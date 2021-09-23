import torch
import torch.nn as nn
from torchvision import models as models


class Encoder(nn.Module):
    def __init__(self, model_path, device):
        super(Encoder, self).__init__()
        self.device = device
        self.vgg = models.vgg16()
        self.vgg.classifier = self.vgg.classifier[:-3]
        self.vgg.load_state_dict(torch.load(model_path, map_location=device))
        self.vgg.to(self.device)

    def forward(self, x):
        return self.vgg(x)
