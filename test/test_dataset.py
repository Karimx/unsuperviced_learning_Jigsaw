
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from processing import ToTensor, ToArray, GridCrop, Shuffle
from data import PuzzleDataset
from processing import perm_subset, toImage, solve
import PIL.Image

transform = Compose([GridCrop(), Shuffle(perm_set=perm_subset(9)), ToTensor()])
transform2 = Compose([ToArray()])
comp = Compose([Shuffle(perm_set=perm_subset(9))])


def test_data():
     train_dataset = PuzzleDataset("../raw-img", transform=transform)
     print(len(train_dataset))
     data_loader = DataLoader(dataset=train_dataset, batch_size=2)
     top = 2
     for i, batch in enumerate(data_loader):
         im_tile, index, perm = batch
         print(len(im_tile))
         print(type(im_tile))

         p = [x[0] for x in perm]
         n = torch.Tensor(p).numpy()
         print(im_tile.shape, "index", index, "perm", n)
         x = toImage(im_tile[0])
         print(x.shape)
         sol_x = toImage(solve(im_tile[0], tuple(n)))
         image = PIL.Image.fromarray(x)
         sol_image = PIL.Image.fromarray(sol_x)
         image.show("None")
         sol_image.show("resolved")

         if i >= top:
            break


def test_dataloader_output():
    train_dataset = PuzzleDataset("../raw-img", transform=transform)
    print(len(train_dataset))
    data_loader = DataLoader(dataset=train_dataset, batch_size=2)
    for i, batch in enumerate(data_loader):
        im_tile, index = batch
        assert im_tile[0] == 2
        assert isinstance(im_tile[1], int)
        break

test_dataloader_output()