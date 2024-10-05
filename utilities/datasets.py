import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from torchvision.ops import masks_to_boxes


import torch


class BBoxDataset(Dataset):
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
            self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform_mask = transforms.Compose([transforms.PILToTensor(), transforms.ToDtype(torch.int64)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.transform(Image.open(self.images[item]))

        label = self.transform_mask(Image.open(self.labels[item]))
        outer = torch.unique(label)[1]
        label = label == outer  # (xmin, ymin, xmax, ymax)
        values = torch.squeeze(masks_to_boxes(label))
        values[2:] -= values[:2]
        return torch.squeeze(image), values


class SegDatasetFromImages(Dataset):
    """
    Custom dataset class for semantic segmentation.

    Args:
        input_images (list): List of paths to input images.
        label_images (list): List of paths to label (mask) images.
        normalize_images (bool): Flag indicating whether to normalize input images.
        ratio: (float): ratio of non cropped images, 0 to 1

    Attributes:
        masks (list): List of paths to label (mask) images.
        images (list): List of paths to input images.
        transform (Compose): Image transformations applied to input images.
        transform_mask (Compose): Image transformations applied to label (mask) images.

    """

    def __init__(self, input_images, label_images, cropped_input=None, cropped_label=None, normalize_images=False,
                 ratio=None):
        self.masks = label_images
        self.images = input_images
        self.cropped_masks = cropped_label
        self.cropped_images = cropped_input
        if cropped_input is None or cropped_label is None or ratio is None:
            self.ratio = 1
        else:
            self.ratio = ratio
        self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)])
        if normalize_images:
            self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.transform_mask = transforms.Compose([transforms.PILToTensor(), transforms.ToDtype(torch.uint8)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if random.random() <= self.ratio:
            image = self.transform(Image.open(self.images[item]))
            label = self.transform_mask(Image.open(self.masks[item]))
        else:
            image = self.transform(Image.open(self.cropped_images[item]))
            label = self.transform_mask(Image.open(self.cropped_masks[item]))
        values = torch.unique(label)
        for idx, i in enumerate(values):
            label[label == i] = idx

        return torch.squeeze(image), torch.squeeze(label)


class SegDatasetFromTensors(Dataset):
    def __init__(self, input_images, label_images, cropped_input=None, cropped_label=None, normalize_images=False,
                 is_training=True, ratio=None):
        self.masks = label_images
        self.images = input_images
        self.cropped_masks = cropped_label
        self.cropped_images = cropped_input
        self.normalize_images = normalize_images
        self.is_training = is_training

        if cropped_input is None or cropped_label is None or ratio is None:
            self.ratio = 1
        else:
            self.ratio = ratio

        if normalize_images:
            self.transform_cropped = CustomNormalize('cropped')
            self.transform_validation = CustomNormalize('validation')
            self.transform_training = CustomNormalize('train')

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, item):
        # Load and normalize the image
        if self.normalize_images:
            if self.is_training:
                if random.random() <= self.ratio:
                    image = self.transform_training(torch.load(self.images[item]) / 255.0)
                    label = torch.load(self.masks[item])
                else:
                    image = self.transform_cropped(torch.load(self.cropped_images[item]) / 255.0)
                    label = torch.load(self.cropped_masks[item])
            else:
                image = self.transform_validation(torch.load(self.images[item]) / 255.0)
                label = torch.load(self.masks[item])
        else:
            if random.random() <= self.ratio:
                image = torch.load(self.images[item]) / 255.0
                label = torch.load(self.masks[item])
            else:
                image = torch.load(self.cropped_images[item]) / 255.0
                label = torch.load(self.cropped_masks[item])

        return torch.squeeze(image), torch.squeeze(label.to(torch.long))


class CustomNormalize(object):
    def __init__(self, dataset_type):
        # Define means and stds for each dataset type
        self.dataset_stats = {
            "validation": {
                "mean": [0.5507591776382487, 0.36368202300487135, 0.28958523739594133],
                "std": [0.17806627200735062, 0.14194286672976278, 0.10991587430013793]
            },
            "cropped": {
                "mean": [0.4685939732421274, 0.28957382473984844, 0.16168409955490093],
                "std": [0.230917657828556, 0.19825984468045332, 0.12745737031473708]
            },
            "train": {
                "mean": [0.31749796040825584, 0.2003174131009406, 0.10739044601664925],
                "std": [0.16542723016378727, 0.13380227445689497, 0.08150097402433934]
            }
        }

        self.mean = torch.tensor(self.dataset_stats[dataset_type]["mean"]).view(-1, 1, 1)
        self.std = torch.tensor(self.dataset_stats[dataset_type]["std"]).view(-1, 1, 1)

    def __call__(self, image):
        # Normalize the entire image
        normalized_image = (image - self.mean) / self.std

        # Create a mask for non-black pixels (black = [0, 0, 0])
        mask = (image > 0).any(dim=0, keepdim=True)  # Non-zero check for all channels

        # Apply the mask to leave black pixels unchanged
        normalized_image = normalized_image * mask + image * (~mask)

        return normalized_image

