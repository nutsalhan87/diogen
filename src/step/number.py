import torch
import torch.nn as nn
import torchvision.models as tm

class NumberDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class NumberNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = tm.resnet50(weights=tm.ResNet50_Weights.DEFAULT)

    def forward(self, x):
        pass

