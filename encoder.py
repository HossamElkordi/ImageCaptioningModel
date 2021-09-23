import argparse
from torchvision import models as models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning Model')
    parser.add_argument('--model', type=str, default='model.pth')
    args = parser.parse_args()

    vgg = models.vgg16()
    vgg.classifier = vgg.classifier[:-3]
    vgg.load_state_dict(args.model)
