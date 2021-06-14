import torch.nn as nn
import torch
from torchvision.models import alexnet as alex, AlexNet


class AlexNet(nn.Module):
    """
        Custom AlexNet like Unsupervised learning paper
    """

    def __init__(self, n_output_payer=100):
        super(AlexNet, self).__init__()
        self.num_tiles = 9
        # Alext net input dim 256×256x3
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )


        self.flatten_feats = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Flatten(0),
        )

        # Not used
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        # Concatenated nineth networks
        self.clasif_concatenated = nn.Sequential(
            nn.Linear(512 * 9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_output_payer)

        )


class JigsawNetAlex(AlexNet):
    """
        Unsupervised learning solving Jigsaw puzzle
        Paper implementation using Alexnet
    """

    def __init__(self, num_tiles):
        """

        Args:
            num_tiles:
            num_perm:
        """
        super().__init__()
        self.tiles = num_tiles


    def forward(self, x):
        """

        Args:
            x: Tiles of dim tensor (B,  tiles, C, W, H)

        Returns:
                float number of index permutation shape (B, 100)
        """
        con = []
        for batch in range(0, x.shape[0]):

            # return (9, 1024)
            tile = self.features(x[batch, :, :, :, :])

            # return (9, 512)
            con.append(self.flatten_feats(tile))

        y = torch.vstack(con)
        y = self.clasif_concatenated(y)
        return y


def jigsawnet_alexnet(tiles=9, pretrained=False) -> torch.nn.Module:
    if pretrained:
        print("Not Implemented")
        return None
    else:
        return JigsawNetAlex(num_tiles=tiles)
