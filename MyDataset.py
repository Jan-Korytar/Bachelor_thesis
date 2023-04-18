import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, input_images, label_images , transform_input= None, transform_label = None):
        self.labels = label_images
        self.images = input_images
        self.transform_input = transform_input
        self.transform_label = transform_label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        label = Image.open(self.labels[item])

        a = '0'

        if self.transform_input is not None:
            image = self.transform_input(image)
        if self.transform_input is not None:
            label = self.transform_label(label)
        return torch.squeeze(image), torch.squeeze(label)
