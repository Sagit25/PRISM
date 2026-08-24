import wandb
import torch
from typing import Dict, Any

class WandbLogger:
    def __init__(self, project: str, name: str, config: Dict[str, Any]):
        wandb.init(project=project, name=name, config=config)

    def log(self, metrics: Dict[str, torch.Tensor], step: int):
        wandb.log({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}, step=step)

    def finish(self):
        wandb.finish()
