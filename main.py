import argparse
import torch
from encoder import Encoder

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='Image Captioning Model')
    parser.add_argument('--model', type=str, default='model.pth')
    parser.add_argument('--out_features', type=str, default='features.json')
    args = parser.parse_args()

    encoderModule = Encoder(model_path=args.model, out_path=args.out_features, device=device)
