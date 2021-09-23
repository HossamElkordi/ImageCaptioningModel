import argparse
import torch
from encoder import Encoder
from ImageParser import ImageDataset, ImageDatasetLoader
from utils import extract_features

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='Image Captioning Model')
    parser.add_argument('--model', type=str, default='model.pth')
    parser.add_argument('--image_folder', type=str, default='Images')
    parser.add_argument('--annotation_file', type=str, default='captions.txt')
    parser.add_argument('--batch_size', type=int, default=50)
    args = parser.parse_args()

    images = ImageDataset(args.image_folder)
    image_loader = ImageDatasetLoader(dataset=images, batch_size=args.batch_size)

    encoderModel = Encoder(model_path=args.model, device=device)
    features = extract_features(model=encoderModel, dataloader=image_loader, device=device)
