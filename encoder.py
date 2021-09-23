import torch
import json
from torchvision import models as models
from tqdm import tqdm


class Encoder():
    def __init__(self, model_path, out_path, device):
        self.device = device
        self.out_path = out_path
        self.vgg = models.vgg16()
        self.vgg.classifier = self.vgg.classifier[:-3]
        self.vgg.load_state_dict(torch.load(model_path, map_location=device))
        self.vgg.to(self.device)

    def extract_features(self, model, dataloader):
        model.eval()
        features_map = {}
        with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
            for it, (ids, images) in enumerate(iter(dataloader)):
                images = images.to(self.device)
                features = model(images)
                for id_i, f_i in enumerate(zip(ids, features)):
                    features_map[id_i] = f_i
            pbar.update()
        json.dump(features_map, open(self.out_path, 'w'))
        return features_map
