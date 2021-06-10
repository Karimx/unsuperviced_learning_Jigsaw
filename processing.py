from typing import Union, Any, List

import torch
import PIL.Image
import numpy as np
from torchvision import transforms
from skimage.util.shape import view_as_blocks


class CenterCrop:
    """
        A custom Center Crop impl to work with ndarrays
    """

    def __init__(self, output_size: tuple, pad: int = 0):
        assert isinstance(output_size, (int, tuple))
        self.pad = pad
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """

        Args:
            image: np.image of shape 3, w, h

        Returns: np.image of shape 3, w, h

        """
        w, h = image.shape[1:3]
        new_w, new_h = self.output_size
        w_pad = (w - new_w) // 2
        h_pad = (h - new_h) // 2
        top = torch.randint(self.pad, self.pad + h_pad, (1,))
        left = torch.randint(self.pad, self.pad + w_pad, (1,))
        image = image[:, top: top + new_h, left: left + new_w]
        return image


class GridCrop:
    """ CenterGridCrop
        Transform over np.array last channel
            Note: Transformation does not work with np by default
    """

    def __init__(self, grid: tuple = (3, 3), center_crop=225, grid_crop: int = 64, channels: int = 3) -> None:
        """

        Args:
            grid: Grid shape ex (3,3)
            center_crop: Center crop
            grid_crop: Final Random crop of each tile
            channels: input image channels
        """
        self.col, self.row = grid
        self.n_tiles = self.col + self.row
        self.center = center_crop
        self.center_crop = transforms.RandomCrop((center_crop, center_crop))
        self.tile_w = center_crop // self.col
        self.tile_h = center_crop // self.row
        self.grid_crop = grid_crop
        self.center_gridcrop = CenterCrop(self.grid_crop, pad=5)

    def __call__(self, sample_image: PIL.Image, transform=None):
        """

        Args:
            sample_image: PIL Image to be converted
            transform: Optional Data augment transformation

        Returns: Tile PIL images of shape T(tiles grid), C, W(center_crop), H(center_crop))

        """
        w, h = sample_image.size
        if h < self.center:
            sample_image = sample_image.resize((w, self.center), resample=PIL.Image.BICUBIC)
        if w < self.center:
            w, h = sample_image.size
            sample_image = sample_image.resize((self.center, h), resample=PIL.Image.BICUBIC)
        img = np.asarray(self.center_crop(sample_image))
        img = img.transpose(2, 1, 0)
        v = view_as_blocks(img, block_shape=(3, self.tile_w, self.tile_h))
        v = np.squeeze(v, 0)
        tiles = np.zeros((9, 3, self.grid_crop, self.grid_crop))
        tile = 0
        for r in range(self.col):
            for c in range(self.row):
                tiles[tile, :, :, :] = self.center_gridcrop(v[r, c, :, :, :])
                tile += 1
        return tiles


class Shuffle:
    """
        Shuffle tiled image for a given permutation set
    """

    def __init__(self, perm_set) -> None:
        self.permutation_ser = perm_set
        # torch.random.manual_seed(1567)

    def __call__(self, im_tiles) -> tuple:
        """ shuffle tile image

        Args:
            im_tiles: input original tiles to be shuffled
                    of dim TILE, CHANNEL, WIDTH, HEIGTH
            permutation: permutation perform on tiles

        Returns: Shuffled permutation using perm set

        """
        random_perm = torch.randint(0, len(self.permutation_ser), (1,))[0]
        p = tuple(self.permutation_ser[random_perm][0])
        shuffled = solve(im_tiles, p)
        return shuffled, self.permutation_ser[random_perm][1]


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self) -> None:
        # Imagenet Normalization
        #  F.normalize(tensor, self.mean, self.std, self.inplace)
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

    def __call__(self, sample) -> tuple:
        return torch.from_numpy(sample[0]).float().div(255), sample[1] #torch.Tensor([sample[1]]).to(torch.int64)


class ToArray:
    """
        PIL Image to numpy array
    """

    def __call__(self, image: PIL.Image) -> np.ndarray:
        arr = np.array(image.resize((225, 255), resample=PIL.Image.BICUBIC))
        # PIL.Image.Image.resize()
        return arr


def solve(im_tiles: np.ndarray, order: tuple):
    """

    Args:
        im_tiles: Unordened tiles of din TILES, CHANNEL, WIDTH, WHEGHT
        order: Permutation of im_tiles

    Returns:

    """
    n_tiles = len(order)
    assert im_tiles.shape[0] == n_tiles
    s = np.empty_like(im_tiles)
    for i in range(len(order)):
        s[i] = im_tiles[order.index(i), :, :, :]
    return s


def perm_subset(n, limit=100):
    """ Valid Permutation subset

    Args:
        n: len of permutation items
        limit:

    Returns: :list Subset tuple (permutation, index (Target))

    """
    perm_set = permute(n)
    subset = []
    for y in range(limit):
        chosen = torch.randint(0, len(perm_set), (1,))[0]
        # chosen = np.random.choice(len(perm_set), size=1)
        subset.append([perm_set[chosen], y])
    return subset


def permute(n: int, limit=100, threshold=None) -> List[tuple]:
    """ Create a permutation subset of a given distance

    Args:
        n:
        limit:
        threshold:

    Returns: permutation

    """
    if threshold is None:
        threshold = 2
    valid_permute = []
    d = int(np.sqrt(n))
    correct = np.array(range(d * d)).reshape((d, d))
    perm = np.copy(correct)
    i = 0
    while i < limit:
        np.random.shuffle(perm)
        if distance(perm, correct) >= threshold:
            t = np.reshape(perm, (n,))
            valid_permute.append(tuple(t))
            i = i + 1
    return valid_permute


def distance(p: np.ndarray, o: np.ndarray) -> int:
    """
        Simple similarity distance
    """
    w, h = p.shape
    total_distance = 0
    for x in range(w):
        for y in range(h):
            if p[x][y] != o[x][y]:
                total_distance = total_distance + 1
    return total_distance


def toImage(img_tile):
    """ Reconstruct tile image to an RGB image

    Args:
        img_tile:

    Returns: ingeter 8 bit rgb image

    """
    tiles, c, w, h = img_tile.shape
    im = np.empty((c, w * 3, h * 3))
    idx = [0, w, w * 2]
    idy = [0, h, h * 2]
    t = 0
    for r in idx:
        for c in idy:
            im[:, r:r + w, c:c + w] = img_tile[t]
            t = t + 1
    im = im.transpose(2, 1, 0)
    im = im * 255
    return im.clip(0, 255).astype(np.uint8)


def toPilImage(img_tile):
    pass
