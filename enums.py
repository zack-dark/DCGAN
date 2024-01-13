from enum import Enum
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

torch.manual_seed(0)


class Param(Enum):
    criterion = nn.BCEWithLogitsLoss()
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.0002

    beta_1 = 0.5
    beta_2 = 0.999
    device = 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


mnist_dataset = MNIST('.', download=True, transform=Param.transform.value)

dataloader = DataLoader(
    mnist_dataset,
    batch_size=Param.batch_size.value,
    shuffle=True
)


