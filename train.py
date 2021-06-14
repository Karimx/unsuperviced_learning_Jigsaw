import os.path
from argparse import ArgumentParser
from typing import Any, Union, List

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning import Callback
import PIL

from jigsawnet import jigsawnet_alexnet
from processing import ToTensor, GridCrop, Shuffle
from processing import perm_subset, toImage, solve
from data import PuzzleDataset
from plot_utils import im_grid


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
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: tiles input of dim (B,T,C,W,H)
                Batch, Tiles, Channels, Width, Height

        Returns: Inference tensor of dim B, 1

        """
        x = self.net.forward(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def training_step(self, train_batch, batch_idx):
        """

        Args:
            train_batch:
            batch_idx:

        Returns: Loss object

        """
        x, y = train_batch
        #x = x.view(x.size(0), -1)
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
        optimizer = torch.optim.Adam(self.net().parameters(), lr=1e-2)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
        #return self.criterion(logits, labels)


class JigsawDataModule(LightningDataModule):
    """
        Wrap Data Module using Lg
    """

    def __init__(self, dataset=None, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.raw_dataset = dataset
        self.train_dataset = None

    def setup(self, stage) -> None:
        """

        Args:
            stage: Train|Test|Val data

        """
        self.train_dataset = PuzzleDataset("raw-img", transform=self.train_transforms)

    def train_dataloader(self):
        """

        Returns: Set Data loader

            Note : pin_memory if dataset fits is RAM or is batch_szie
            Note : num_workes need more mem and pag file size
        """
        return DataLoader(dataset=self.train_dataset, num_workers=4, batch_size=4, pin_memory=True)


class FeatureCallback(Callback):
    """ PL Callback to save feats image"""

    def __init__(self):
        super().__init__()
        self.metrics = []
        self.epoch = 0

    def on_epoch_start(self, trainer, pl_module) -> None:
        tensor = pl_module.net.features[0].weight.clone().detach().cpu()
        print("Mean filter", torch.mean(tensor))
        im_name = os.path.join("lightning_logs", f"features_epoch_{self.epoch}.jpg")
        im = im_grid(tensor)
        img = PIL.Image.fromarray(im)
        img.save(im_name)
        self.epoch += 1
        print("filters image mean", im.mean())

    def on_epoch_end(self, trainer, pl_module):
        tensor = pl_module.net.features[0].weight.clone().detach().cpu().numpy()
        print("Mean filter END", tensor.mean())



def main(hparams):
    """

    Args:
        hparams:

    Returns: None

    """
    m = jigsawnet_alexnet(9)
    model = LitJigSaw(m)
    train_transform = Compose([GridCrop(), Shuffle(perm_set=perm_subset(9)), ToTensor()])
    train_dataset = JigsawDataModule(hparams.train_dataset, train_transforms=train_transform)
    # time 6.0 -> 4.2. -> 3.45 -> 3.20
    # precision=16, -0.06873874
    trainer = Trainer(gpus=1, callbacks=[FeatureCallback()])
    trainer.fit(model, train_dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--tile_grid', default=3)
    parser.add_argument('--train_dataset', default="raw-img")
    args = parser.parse_args()

    main(args)
