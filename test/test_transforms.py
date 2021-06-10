import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.util.shape import view_as_blocks
from torchvision import transforms
from skimage.util import crop

from processing import GridCrop, permute

CENTER_CROP = 225
GRID = (3, 3)
TILE = 75
INFERENCE_TILE = 65


# tensor_test = torch.randn()


def shuffle_class_test():
    pass


def gridcrop_class_test():
    grid_transform = GridCrop(GRID, 225)

    im_sample = np.random.randn(400, 400, 3).clip(0, 1)
    im = Image.fromarray((im_sample* 255).astype('uint8'))
    new = grid_transform(im)
    print(new.shape)



def permute_test():
    elements = 6
    result = permute(4, elements)
    assert len(result) == elements


# permute_test()
gridcrop_class_test()
