import torch
import torch.nn as nn
import torchvision.models as tm


class NumberDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


class NumberNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = tm.resnet18()

    def forward(self, x):
        pass
