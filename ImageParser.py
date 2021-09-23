import os
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, img_folder):
        super(ImageDataset, self).__init__()
        data_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        self.data = []
        for file in os.listdir(img_folder):
            img = data_transform(mpimg.imread(os.path.join(img_folder, file)))
            img.unsqueeze(0)
            self.data.append((file.split('.')[0], img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ImageDatasetLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super(ImageDatasetLoader, self).__init__(dataset=dataset, batch_size=batch_size)
