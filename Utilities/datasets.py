import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

class BboxDataset(Dataset):
    """
    Custom dataset class for bounding box detection.

    Args:
        input_images (list): List of paths to input images.
        label_images (list): List of paths to label (mask) images.
        normalize_images (bool): Flag indicating whether to normalize input images.

    Attributes:
        labels (list): List of paths to label (mask) images.
        images (list): List of paths to input images.
        transform (Compose): Image transformations applied to input images.
        transform_mask (Compose): Image transformations applied to label (mask) images.

    """
    def __init__(self, input_images, label_images, normalize_images):
        self.labels = label_images
        self.images = input_images
        self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)])
        if normalize_images:
            self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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


class SegDataset(Dataset):
    """
    Custom dataset class for semantic segmentation.

    Args:
        input_images (list): List of paths to input images.
        label_images (list): List of paths to label (mask) images.
        normalize_images (bool): Flag indicating whether to normalize input images.

    Attributes:
        labels (list): List of paths to label (mask) images.
        images (list): List of paths to input images.
        transform (Compose): Image transformations applied to input images.
        transform_mask (Compose): Image transformations applied to label (mask) images.

    """
    def __init__(self, input_images, label_images, normalize_images):
        self.labels = label_images
        self.images = input_images
        self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)])
        if normalize_images:
            self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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
