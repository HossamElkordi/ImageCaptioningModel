import torch, torchvision
from torchvision import models as models


Resnet = models.resnet101(pretrained=True, progress=True)
