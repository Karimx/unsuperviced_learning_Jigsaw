from argparse import ArgumentParser
from typing import Any, Union, List

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.core.optimizer import LightningOptimizer

from jigsawnet import jigsawnet_alexnet
from processing import ToTensor, GridCrop, Shuffle
from processing import perm_subset, toImage, solve
from data import PuzzleDataset


class LitJigSaw(LightningModule):

    def __init__(self, net: nn.Module) -> None:
        """
            Lighthing model system
        Args:
            *args:
            **kwargs:
        """
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> Any:
        """

        Args:
            x: tiles input of dim (B,T,C,W,H)
                Batch, Tiles, Channels, Width, Height

        Returns: Inference tensor of dim B, 1

        """
        x = self.net(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def training_step(self, train_batch, batch_idx):
        """

        Args:
            train_batch:
            batch_idx:

        Returns:

        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        """

        Args:
            use_pl_optimizer:

        Returns:

        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)


class JigsawDataModule(LightningDataModule):
    """
        Wrap Data Module using Lg
    """

    def setup(self, stage):
        train_transform = Compose([GridCrop(), Shuffle(perm_set=perm_subset(9)), ToTensor()])
        self.train_dataset = PuzzleDataset("raw-img", transform=train_transform)


    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, num_workers=8, batch_size=4)


def main(hparams):
    """

    Args:
        hparams:

    Returns: None

    """
    m = jigsawnet_alexnet(9)
    model = LitJigSaw(m)
    train_dataset = JigsawDataModule()
    trainer = Trainer(gpus=1)
    trainer.fit(model, train_dataset)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--tile_grid', default=3)
    args = parser.parse_args()

    main(args)
