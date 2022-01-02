from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchaudio.transforms import MuLawEncoding, MuLawDecoding


import model as module_arch
import model.condition as module_condition
import datasets as module_data
from utils import get_instance


class LightModel(pl.LightningModule):
    model: nn.Module
    conditioner: nn.Module
    criterion: nn.Module

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('Lightning')
        parser.add_argument('--q_channels', type=int, default=256)
        parser.add_argument('--padding_method', type=str,
                            choices=['same', 'valid', 'cache'], default='same')
        return parent_parser

    def __init__(self, q_channels: int, padding_method: str,  config_dict: dict = None, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(config_dict)
        self.save_hyperparameters("q_channels", "padding_method")

        model = get_instance(module_arch, self.hparams.arch)
        conditioner = get_instance(module_condition, self.hparams.conditioner)

        self.model = model
        self.conditioner = conditioner
        self.enc = MuLawEncoding(self.hparams.q_channels)
        self.dec = MuLawDecoding(self.hparams.q_channels)

    def configure_optimizers(self):
        optimizer = get_instance(
            torch.optim, self.hparams.optimizer, self.parameters())
        return optimizer

    def train_dataloader(self):
        train_data = get_instance(module_data, self.hparams.dataset)
        train_loader = DataLoader(
            train_data, **self.hparams.data_loader)
        return train_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        cond = self.conditioner(y)
        x = self.enc(x)
        y = self.enc(y)
        pred = self.model(
            x, cond, True if self.hparams.padding_method == 'same' else False)
        y = y[:, -pred.shape[-1]:]
        loss = F.cross_entropy(pred.unsqueeze(-1), y.unsqueeze(-1))
        return loss
