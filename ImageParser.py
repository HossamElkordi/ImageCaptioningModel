import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, img_folder):
        super(ImageDataset, self).__init__()
        data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.data = []
        print('Loading Images...')
        for file in os.listdir(img_folder):
            img = data_transform(cv2.imread(filename=os.path.join(img_folder, file)))
            img = torch.unsqueeze(input=img, dim=0)
            self.data.append((file.split('.')[0], img))
        print('Done')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ImageDatasetLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super(ImageDatasetLoader, self).__init__(dataset=dataset, batch_size=batch_size)
