
from PIL import Image
import numpy as np

def plot_puzzle(puzzle):
    Image.fromarray(puzzle).show()


def reconstruct_image_from_puzzle(label, sub_images, read_from_pickle=False):
    if isinstance(sub_images, str) and not read_from_pickle:
        raise ("Please set read_from_pickle to True")

    if read_from_pickle:
        sub_images = read_sub_images(sub_images)
    num_sub_images = len(sub_images)
    indices = get_permutation_from_label(label, num_sub_images)
    reconstruction = []
    for i in range(len(indices)):
        position = indices.index(i)
        reconstruction.append(sub_images[position])
    puzzle = make_puzzle(reconstruction, rows=int(sqrt(num_sub_images)), cols=int(sqrt(num_sub_images)))
    # TODO tie f
    # plot_puzzle(puzzle)


def reconstruct_image(image: np.array, perm: set) -> None:
    """

    :param image: tiles of shape (4 tiles,channel 3,tile h,tile w)
    :param perm: permutation list example: (1, 3, 4 , 2)
    :return:
    """
    tiles = image.shape[0]
    width = image.shape[2]
    height = image.shape[3]
    assert tiles == len(perm)
    #raise "Not equals permutations"
    start = 0

    for index in perm:
        np.array().transpose()
        t = tuple([1,2,3])
        t.index()
        s = set([1,2,3])




