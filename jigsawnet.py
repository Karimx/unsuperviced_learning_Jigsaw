import torch.nn as nn
import torch
from torchvision.models import alexnet as alex, AlexNet


class Jigsaw(nn.Module):

    def __init__(self, num_tiles, num_outputs):
        super(Jigsaw, self).__init__()

        self.num_tiles = num_tiles
        self.num_outputs = num_outputs
        # self.normalization = nn.Sequential(Normalization())
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.extract_features = nn.Sequential(
            nn.Conv2d(1, 32, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(inplace=True),
            )
        self.latent_space = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4 * 4 * 32, 512),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512 * num_tiles, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, self.num_outputs),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):

        for i in range(self.num_tiles):
            x.cat_(self.process_per_tile(x[:, i, :, :, :]), 1)

        x = self.classifier(x)
        return x

    def process_per_tile(self, x):
        x = self.extract_features(x)
        x = self.avgpool(x)
        x = x.view((-1, x.size()[1] * x.size()[2] * x.size()[3]))
        x = self.latent_space(x)
        return x


class AlexNet(nn.Module):
    """
        Custom AlexNet like Unsupervised learning paper
    """

    def __init__(self, num_tiles=9, n_output_payer=100):
        super(AlexNet, self).__init__()

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
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        # Original nn.Linear(256 * 6 * 6, 4096) = 9216, 4096
        # Paper implementation                   (512 * 9, ,(4608, 4096)
        # Concatenated nineth networks
        self.clasif_concatenated = nn.Sequential(
            nn.Dropout(),
            #nn.Linear(512 * num_tiles, 4096),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_output_payer),
            # nn.ReLU(inplace=True),
            # Note: paper implementation 100 - > 64
            # nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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
        super().__init__(num_tiles)
        self.tiles = num_tiles
        #self.num_perm = num_perm

    def forward(self, x):
        """

        Args:
            x: Tiles of dim tensor (N,  tiles, C, W, H)

        Returns:
                float number of index permutation
        """
        n = x.shape[0]
        fc7_concat_feats = []
        for t in range(self.tiles):
            tile_infer = torch.flatten(self.avgpool(self.features(x[:, t, :, :, :])), 0)
            tile_infer.unsqueeze_(0)
            fc7_concat_feats.append(tile_infer)

        y = torch.cat(fc7_concat_feats, 1)
        y = y.view(n, y.shape[1]//n)
        return self.clasif_concatenated(y)


def jigsawnet_alexnet(tiles= 9, pretrained=False) -> torch.nn.Module:
    if pretrained:
        print ("Not Implemented")
        return None
    else:
        return JigsawNetAlex(num_tiles=tiles)
