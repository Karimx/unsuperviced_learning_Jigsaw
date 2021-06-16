import PIL.Image
import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

from processing import solve, toImage


def im_puzzle(puzzle: np.ndarray, im_perm: tuple) -> PIL.Image:
    """

    Args:
        im_perm:
        puzzle: shape T,C,W,H

    Returns: PIL image

    """
    sol_x = toImage(solve(puzzle, im_perm))
    return Image.fromarray(puzzle)


def im_grid(features: torch.Tensor) -> np.ndarray:
    """

    Args:
        features: Grid Tensor of shape N,C,H,W

    Returns: Numpy image

        Note: can use 'from skimage import img_as_ubyte'

        Examples:
            im_grid(alex_model.features[0].weight)
    """
    assert len(features.shape) == 4
    filters = make_grid(features).numpy()
    filters += np.abs(filters.min())
    filters *= (255.0 / filters.max())
    filters = filters.clip(0, 255).astype(np.uint8)
    return np.transpose(filters, (2, 1, 0))




