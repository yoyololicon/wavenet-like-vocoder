from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchaudio.transforms import MuLawEncoding, MuLawDecoding


import model as module_arch
from model.base import MemModel
import model.condition as module_condition
import model.upsample as module_upsample
import datasets as module_data
from utils import get_instance

class LightModel(pl.LightningModule):
    model: MemModel
    conditioner: nn.Module
    upsampler: nn.Module
    criterion: nn.Module
    emb: nn.Sequential

    def __init__(self,
                 cfg: dict = None,
                 **kwargs) -> None:
        super().__init__()
        # converting omegaconf into python dict
        # I usually don't use save_hyperparameters
        # since hydra already helps you to save a copy of the copy after each run   
        self.cfg = cfg
        
        if self.cfg.memory_segment is not None:
            self.automatic_optimization = False

        model = get_instance(module_arch, self.cfg.arch)
        conditioner = get_instance(module_condition, self.cfg.conditioner)
        upsampler = get_instance(module_upsample, self.cfg.upsampler)

        self.model = model
        self.conditioner = conditioner
        self.upsampler = upsampler
        self.emb = nn.Sequential(nn.Embedding(self.cfg.q_channels, self.cfg.emb_channels),
                                 nn.Tanh())
        self.enc = MuLawEncoding(self.cfg.q_channels)
        self.dec = MuLawDecoding(self.cfg.q_channels)

    def configure_optimizers(self):
        optimizer = get_instance(
            torch.optim, self.cfg.optimizer, self.parameters())
        return optimizer

    def train_dataloader(self):
        train_data = get_instance(module_data, self.cfg.dataset)
        train_loader = DataLoader(
            train_data, **self.cfg.data_loader)
        return train_loader

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            return self._manual_training_step(batch)
        x, y = batch
        cond = self.upsampler(self.conditioner(y))[..., :x.shape[-1]]
        x = self.enc(x)
        y = self.enc(y)
        x = self.emb(x).transpose(1, 2)
        pred = self.model(
            x, cond, zeropad=True if self.cfg.padding_method == 'same' else False)
        y = y[:, -pred.shape[-1]:]
        loss = F.cross_entropy(pred.unsqueeze(-1), y.unsqueeze(-1))
        self.log('loss', loss, prog_bar=False, on_step=True)
        return loss

    def _manual_training_step(self, batch):
        x, y = batch
        cond = self.upsampler(self.conditioner(y))[..., :x.shape[-1]]
        x = self.enc(x)
        y = self.enc(y)
        assert x.shape[-1] == y.shape[-1] == cond.shape[-1]
        total_size = y.numel()
        memories = self.model.init_memories(x.size(0), device=x.device)
        chunk_size = self.cfg.memory_segment

        opt = self.optimizers()
        opt.zero_grad()
        total_loss = 0
        for sub_x, sub_y, sub_cond in zip(x.split(chunk_size, -1), y.split(chunk_size, -1), cond.split(chunk_size, -1)):
            sub_x = self.emb(sub_x).transpose(1, 2)
            pred = self.model(
                sub_x, sub_cond, memories=memories, zeropad=False)
            loss = F.cross_entropy(
                pred.unsqueeze(-1), sub_y.unsqueeze(-1), reduction='sum') / total_size
            self.manual_backward(loss)
            total_loss += loss.item()
        opt.step()

        self.log('loss', total_loss, prog_bar=True, on_step=True)
        return loss