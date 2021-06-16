import glob

from PIL import Image
import numpy as  np
from torch.utils.data import Dataset

from processing import Shuffle, ToTensor, perm_subset


class PuzzleDataset(Dataset):
    """
        Dataset
    """

    def __init__(self, path_dataset, grid_split=9, per_image=32, transform=None):
        self.path_dataset = path_dataset
        self.transform = transform
        self.files = glob.glob(f"{self.path_dataset}/*/*.jpeg", recursive=True)
        self.len = len(self.files)
        self.grid_split = grid_split
        self.permute_Xy = tuple(perm_subset(grid_split))
        self.shuflle_tiles = Shuffle(perm_set=self.permute_Xy)
        self.current_index = 0
        self.n_shuffle = per_image
        self.tenrorize = ToTensor()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self.current_index = index
        im_path = self.files[self.current_index]
        img = Image.open(im_path)
        r = self.transform(img)

        return self.get_transforms(r)

    def get_transforms(self, tiles: np.ndarray):
        """

        Args:
            tiles: Single tile image

        Returns:

        """
        print("image", self.files[self.current_index])

        list_tile = []
        for t in range(0, self.n_shuffle):
            X, y = self.shuflle_tiles(tiles)
            print(X.shape, y)
            list_tile.append(self.tenrorize((X, y)))

        return list_tile


