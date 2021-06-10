import torch

from jigsawnet import JigsawNetAlex

model_defaults = {
    "num_tiles": 9,
    "num_perm": 100,
}


def test_output():
    model = JigsawNetAlex(**model_defaults)
    tensor = torch.randn((1, 9, 3, 64, 64))
    print(tensor.shape)
    result = model(tensor)
    print(result)
    print(result.shape)


def test_im_output():
    model = JigsawNetAlex()


test_output()
