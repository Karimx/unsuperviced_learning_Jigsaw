import numpy  as np

from image_utils import Puzzle
from preprocessing import get_sub_images
from processing import permute


def test_premutation():
    permute = Puzzle((10, 10), 4)
    arr = np.array([range(3 * 10 * 10)])
    arr = arr.reshape((3, 10, 10))
    indexes = permute.start_xy_index()
    print(indexes)
    tiles = permute.split(arr)
    print(f"{tiles.shape}")


def test_get_subimages():
    w = 10
    h = 10
    puzzle_pieces = 4
    p = Puzzle(size=(10, 10), pieces=puzzle_pieces)
    arr = np.array([range(3 * w * h)])
    arr = arr.reshape((w, h, 3))
    tiles = get_sub_images(arr, 2, 2)
    print(tiles.shape)
    assert tiles.shape == (puzzle_pieces, w // 2, h // 2, 3)


print(permute(4, 5))

# chosee()
