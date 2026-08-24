# PRISM: Physics-guided Refraction-aware Inverse Scene Matting

This repository contains the official implementation of **PRISM (Physics-guided Refraction-aware Inverse Scene Matting)**, as proposed in Seoul National University (August 2026). 

PRISM jointly estimates a reusable counterfactual background ($B^{cf}$) and a per-frame colored refractive transparent foreground operator from a single fixed-camera video of a moving transparent object without requiring a clean plate.

---

## 📂 Repository Structure

*   **`RCDatasetCreation/`**: Forked from RCTrans dataset generator. Implements the physics-controlled synthetic video generator using Mitsuba 3 ray tracing with fixed camera/light and moving transparent objects.
*   **`network/`**: Core neural network codebase. Integrates Facebook's official SAM2.1 with a clean-room reproduction of MAM2 (PDD & MSS), Hiera LoRA injection, Shared Background Reconstruction modules, Physics Matter Heads, and a unified training/testing interface.

---

## 🚀 Execution Guide (How to Run)

Ensure that your environment matches the requirements (Python 3.10+, PyTorch 2.5.1+, Mitsuba 3.7.0, and wandb).

### 1. Dataset Generation (`RCDatasetCreation`)
Before training, you must generate the synthetic dataset. 

*   **CPU Smoke Test (96x96 watertight sphere):**
    ```bash
    cd RCDatasetCreation
    python render_dataset.py --conf configs/dataset_prism_main_smoke.yaml
    ```
*   **Full Main Training Set (Requires GPU with CUDA Mitsuba variant):**
    ```bash
    cd RCDatasetCreation
    python render_dataset.py --conf configs/dataset_prism_main.yaml
    ```
*   **Diagnostic Split (Fresnel reflection, caustics, and shadows enabled):**
    ```bash
    cd RCDatasetCreation
    python render_dataset.py --conf configs/dataset_prism_diagnostic_reflection.yaml
    ```

---

### 2. Unified Training & Evaluation Pipeline (`network`)
We have implemented a unified execution script in `network/src/train.py`. It integrates both the training loop and testing phase, complete with online **WandB (Weights & Biases)** tracking.

To run, export the project root to `PYTHONPATH` and execute `train.py`:

```bash
# 1. Set environment variables
export PYTHONPATH=$PYTHONPATH:/Users/yangsukhun/PRISM

# 2. Login to wandb (if not logged in)
wandb login

# 3. Run Training & Evaluation (Mode: both)
python network/src/train.py \
    --train_data /Users/yangsukhun/PRISM/RCDatasetCreation/result/prism_main_smoke/train \
    --test_data /Users/yangsukhun/PRISM/RCDatasetCreation/result/prism_main_smoke/test \
    --project_name PRISM-Full-Experiment \
    --epochs 10 \
    --lr 1e-4 \
    --mode both \
    --save_dir ./checkpoints
```

#### Command Arguments:
*   `--train_data`: Path to the generated training dataset.
*   `--test_data`: Path to the generated test/validation dataset.
*   `--project_name`: Your WandB project name.
*   `--epochs`: Number of training epochs.
*   `--lr`: Learning rate (Adam optimizer).
*   `--mode`: Execution mode (`train` for training only, `test` for test evaluation only, `both` for sequence training and final testing).
*   `--save_dir`: Directory to save model checkpoint `.pth` files.
*   `--checkpoint`: Path to load an existing checkpoint (mandatory for `--mode test`).

---

## 🔍 Code Review & Roadmap of Remaining Work

The core pipeline (MAM2 + PDD + MSS + Refractive Splatting + Robust Fusion) has been completely implemented and mathematically validated against the strict PRISM physical contract. However, to complete the research and produce publishable experimental results, the following tasks must be completed:

### Task 1: Full-scale GPU Dataset Generation & Validation
*   **Current State:** Verified via a CPU-based 96x96 smoke test.
*   **To Do:** Execute rendering on a CUDA-supported GPU environment with `dataset_prism_main.yaml` to create high-resolution sequence data. Validate that the generated outputs strictly conform to the identity and factorization constraints of the canonical PRISM contract.

### Task 2: Stage-1 Semantic Weight Pre-training (MAM2)
*   **Current State:** The extension weights for `PDD` and `MSS` are integrated and zero-initialized.
*   **To Do:** Perform Stage-1 training using VOS (Video Object Segmentation) and classical image matting datasets to pre-train the semantic boundaries and trimap outputs before connecting them to physics-guided optimization.

### Task 3: Joint Optimization Hyperparameter & Loss Tuning
*   **Current State:** `losses.py` implements all loss terms (semantic, component, background, rendering, and reusability consistency $\mathcal{L}_{reuse}$).
*   **To Do:** Run the Stage-3 joint training script and tune the loss weights ($\lambda_{sem}$, $\lambda_{op}$, $\lambda_{bg}$, $\lambda_{render}$, $\lambda_{reg}$, $\lambda_{reuse}$) to ensure stable gradient propagation through the unrolled fixed-point iterations without vanishing or exploding gradients.

### Task 4: Baseline Comparison & Metric Evaluation
*   **Current State:** Unified test script calculates $\alpha$ SAD/MSE and background reconstruction MSE.
*   **To Do:** Produce benchmark results comparing PRISM against:
    1.  *Foreground operators:* TOM-Net, CTOM-Net, TransMatting.
    2.  *Inpainting/Background:* robust aggregation (median), STTN, ProPainter, DiffuEraser.
    3.  *Joint decomposition:* Omnimatte, OmnimatteRF.

### Task 5: 3D Geometry Recovery (Extension Phase)
*   **Current State:** Marked as "Planned" in `Table 3`.
*   **To Do:** Once the 2D refractive flow $u$ is stable, implement the depth-normal prior optimization from Section 8.3 of the proposal using Snell's law ray tracing to recover 3D shapes.
