import glob

from PIL import Image
from torch.utils.data import Dataset

from processing import perm_subset


class PuzzleDataset(Dataset):
    """
        Dataset
    """

    def __init__(self, path_dataset, grid_split=9, transform=None):
        self.path_dataset = path_dataset
        self.transform = transform
        self.files = glob.glob(f"{self.path_dataset}/*/*.jpeg", recursive=True)
        self.len = len(self.files)
        self.labels = []
        self.grid_split = grid_split
        self.permute_Xy = perm_subset(self.grid_split)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        im_path = self.files[index]
        img = Image.open(im_path)
        r = self.transform(img)
        return r[0], r[1]
