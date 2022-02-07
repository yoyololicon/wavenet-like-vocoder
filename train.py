import os
import json
import argparse

import torch


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


from model import LightModel

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


class ChangeLRCallback(pl.Callback):
    def __init__(self, lr: float) -> None:
        super().__init__()
        self.lr = lr

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr

@hydra.main(config_path="configs", config_name="waveformer_eva_128_2048")
def main(cfg):
    pl.seed_everything(cfg.seed)
    
    # convert relative path to absolute path
    cfg.dataset.args.data_dir = to_absolute_path(cfg.dataset.args.data_dir)
    
    experiment_name = f"{cfg.arch.type}-memory_segment{cfg.memory_segment}"

    gpus = torch.cuda.device_count()
    if cfg is not None:
        cfg.data_loader.batch_size //= gpus

    callbacks = [
        ModelSummary(max_depth=-1),
        LearningRateMonitor('epoch'),
        ModelCheckpoint(save_top_k=-1)
    ]

    if cfg.lr:
        callbacks.append(ChangeLRCallback(cfg.lr))

    if cfg.ckpt_path:
        lit_model = LightModel.load_from_checkpoint(cfg.ckpt_path)
    else:
        lit_model = LightModel(cfg=cfg)

    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)        
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        gpus=gpus,
        strategy=DDPPlugin(find_unused_parameters=False) if gpus > 1 else None,
        logger=logger 
    )
        
    trainer.fit(lit_model, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    main()

