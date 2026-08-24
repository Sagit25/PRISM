import torch
import torch.optim as optim
from pathlib import Path
from network.src.dataset import RCTransPRISMDataset
from network.src.pipeline import RefractiveMAM2
from network.src.sam2_integration import MAM2VideoPredictor
from network.src.training import (
    configure_stage1,
    configure_stage2,
    configure_joint,
    selective_semantic_loss,
    physics_stage_loss,
    joint_stage_loss,
    SemanticTargets,
)
from network.src.logger import WandbLogger
from network.src.config import PipelineConfig
from network.src.losses import RefractiveGroundTruth
import argparse
import os

def evaluate(pipeline, dataloader, device, stage=3):
    pipeline.eval()
    total_metrics = {}
    
    with torch.no_grad():
        for sample in dataloader:
            tensors = {k: v.to(device) for k, v in sample["tensors"].items()}
            batch_frames = tensors.pop("frames")
            target = RefractiveGroundTruth(**tensors)
            output = pipeline(batch_frames)
            
            # Select appropriate evaluation loss based on stage
            if stage == 1:
                sem_targets = SemanticTargets(object_mask=target.object_mask, trimap=target.trimap)
                loss_terms = selective_semantic_loss(
                    output.backbone.mask_logits,
                    output.backbone.trimap_logits,
                    sem_targets,
                    dataset_kind="synthetic_physics"
                )
            elif stage == 2:
                loss_terms = physics_stage_loss(output, target)
            else:
                loss_terms = joint_stage_loss(output, target)
                
            for k, v in loss_terms.items():
                total_metrics[k] = total_metrics.get(k, 0) + v.item()
                
            # Additional explicit metrics evaluation (SAD, MSE for alpha and background)
            if stage in [2, 3]:
                if target.alpha is not None:
                    alpha_diff = (output.matter.alpha - target.alpha).abs()
                    total_metrics["eval_alpha_SAD"] = total_metrics.get("eval_alpha_SAD", 0) + alpha_diff.sum().item()
                    total_metrics["eval_alpha_MSE"] = total_metrics.get("eval_alpha_MSE", 0) + (alpha_diff ** 2).mean().item()
                
                if target.counterfactual_background is not None:
                    bg_gt = target.counterfactual_background[:, None].expand(-1, batch_frames.shape[1], -1, -1, -1) if target.counterfactual_background.ndim == 4 else target.counterfactual_background
                    bg_diff = (output.background.background - bg_gt[:, 0]).abs()
                    total_metrics["eval_bg_MSE"] = total_metrics.get("eval_bg_MSE", 0) + (bg_diff ** 2).mean().item()

    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}
    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description="PRISM Unified Stage-by-Stage Training Pipeline")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training subset")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test subset")
    parser.add_argument("--project_name", type=str, default="PRISM-Project", help="W&B Project name")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both", help="Execution mode")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3, help="Training Stage (1: Semantic, 2: Physics-only, 3: Joint)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for testing or starting stage")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu")

    # 1. Datasets & Dataloaders
    train_dataset = RCTransPRISMDataset(root=Path(args.train_data), strict_contract=False)
    test_dataset = RCTransPRISMDataset(root=Path(args.test_data), strict_contract=False)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. Model Init
    config = PipelineConfig()
    predictor = MAM2VideoPredictor() 
    pipeline = RefractiveMAM2(backbone=predictor, config=config)

    # Load existing checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        pipeline.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 3. W&B Logger Init
    logger = WandbLogger(project=args.project_name, name=f"PRISM-Stage{args.stage}-{args.mode}", config=vars(args))

    # 4. Training Mode
    if args.mode in ["train", "both"]:
        # Configure trainable parameters and pipeline states based on selected stage
        if args.stage == 1:
            print(">>> Configuring Stage 1: Semantic (PDD/MSS & LoRA) Training")
            params = configure_stage1(predictor)
        elif args.stage == 2:
            print(">>> Configuring Stage 2: Physics-only (Decomposition & Completion) Training")
            params = configure_stage2(predictor, pipeline)
        else:
            print(">>> Configuring Stage 3: Joint (Semantic + Physics) Training")
            params = configure_joint(predictor, pipeline)

        optimizer = optim.Adam(params, lr=args.lr)
        print(f">>> Starting Stage {args.stage} Training Phase...")
        
        global_step = 0
        for epoch in range(args.epochs):
            for i, sample in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                tensors = {k: v.to(device) for k, v in sample["tensors"].items()}
                batch_frames = tensors.pop("frames")
                target = RefractiveGroundTruth(**tensors)
                
                output = pipeline(batch_frames)
                
                # Compute stage-specific loss
                if args.stage == 1:
                    sem_targets = SemanticTargets(object_mask=target.object_mask, trimap=target.trimap)
                    loss_terms = selective_semantic_loss(
                        output.backbone.mask_logits,
                        output.backbone.trimap_logits,
                        sem_targets,
                        dataset_kind="synthetic_physics"
                    )
                elif args.stage == 2:
                    loss_terms = physics_stage_loss(output, target)
                else:
                    loss_terms = joint_stage_loss(output, target)
                
                loss_terms["total"].backward()
                optimizer.step()
                
                # Log step loss
                logger.log(loss_terms, step=global_step)
                global_step += 1
                
                if i % 10 == 0:
                    print(f"[Stage {args.stage} | Epoch {epoch}/{args.epochs}] Step {i}, Loss: {loss_terms['total'].item():.6f}")

            # Inline epoch validation
            eval_metrics = evaluate(pipeline, test_dataloader, device, stage=args.stage)
            print(f"[Stage {args.stage} | Epoch {epoch} Eval] metrics: {eval_metrics}")
            logger.log({f"val_{k}": v for k, v in eval_metrics.items()}, step=global_step)
            
            # Save checkpoints
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_stage{args.stage}_epoch_{epoch}.pth")
            torch.save(pipeline.state_dict(), checkpoint_path)
            
            # Reset pipeline states to train mode after evaluation
            if args.stage == 1:
                configure_stage1(predictor)
            elif args.stage == 2:
                configure_stage2(predictor, pipeline)
            else:
                configure_joint(predictor, pipeline)
        
        # Save final checkpoint
        final_ckpt = os.path.join(args.save_dir, f"checkpoint_stage{args.stage}_final.pth")
        torch.save(pipeline.state_dict(), final_ckpt)
        args.checkpoint = final_ckpt

    # 5. Testing Mode
    if args.mode in ["test", "both"]:
        print(f">>> Starting Stage {args.stage} Evaluation Phase...")
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            pipeline.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        test_metrics = evaluate(pipeline, test_dataloader, device, stage=args.stage, is_final_test=True)
        print(f"\n================== FINAL STAGE {args.stage} TEST METRICS ==================")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.6f}")
        print("========================================================\n")
        
        logger.log({f"test_{k}": v for k, v in test_metrics.items()}, step=99999)

    logger.finish()
    print("Execution Finished Successfully.")

if __name__ == "__main__":
    main()
