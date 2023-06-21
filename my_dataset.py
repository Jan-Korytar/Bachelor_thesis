import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class MyDataset(Dataset):
    def __init__(self, input_images, label_images):
        self.labels = label_images
        self.images = input_images
        self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize( [0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])])
        self.transform_mask = transforms.Compose([transforms.PILToTensor(), transforms.ToDtype(torch.int64)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.transform(Image.open(self.images[item]))
        label = self.transform_mask(Image.open(self.labels[item]))
        values = torch.unique(label)
        for idx, i in enumerate(values):
            label[label == i] = idx

        return torch.squeeze(image), torch.squeeze(label)
