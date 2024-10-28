import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2 as transforms


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





class SegDatasetFromTensors(Dataset):
    def __init__(self, input_img, masks, cropped_input=None, cropped_masks=None, normalize_images=True,
                 is_training=True, ratio=None):
        self.masks = masks
        self.images = input_img
        self.cropped_masks = cropped_masks
        self.cropped_images = cropped_input
        self.normalize_images = normalize_images
        self.is_training = is_training

        if cropped_input is None or cropped_masks is None or ratio is None:
            self.ratio = 1
        else:
            self.ratio = ratio

        if normalize_images:
            self.transform_validation_cropped = CustomNormalize('validation_cropped')
            self.transform_training_cropped = CustomNormalize('train_cropped')
            self.transform_validation = CustomNormalize('validation')
            self.transform_training = CustomNormalize('train')

    def __len__(self):
        if self.masks:
            return len(self.masks)
        else:
            return len(self.cropped_masks)

    def __getitem__(self, item):
        # Load and normalize the image
        if self.normalize_images:
            if self.is_training:
                if random.random() <= self.ratio:
                    image = self.transform_training(torch.load(self.images[item], weights_only=False) / 255.0)
                    label = torch.load(self.masks[item], weights_only=False)
                else:
                    image = self.transform_training_cropped(
                        torch.load(self.cropped_images[item], weights_only=False) / 255.0)
                    label = torch.load(self.cropped_masks[item], weights_only=False)
            else:
                if random.random() <= self.ratio:
                    image = self.transform_validation(torch.load(self.images[item], weights_only=False) / 255.0)
                    label = torch.load(self.masks[item], weights_only=False)
                else:
                    image = self.transform_validation_cropped(
                        torch.load(self.cropped_images[item], weights_only=False) / 255.0)
                    label = torch.load(self.cropped_masks[item], weights_only=False)
        else:
            if random.random() <= self.ratio:
                image = torch.load(self.images[item], weights_only=False) / 255.0
                label = torch.load(self.masks[item], weights_only=False)
            else:
                image = torch.load(self.cropped_images[item], weights_only=False) / 255.0
                label = torch.load(self.cropped_masks[item], weights_only=False)

        return torch.squeeze(image), torch.squeeze(label.to(torch.long))


class PredictionDataset(Dataset):
    """
        A custom PyTorch dataset for handling image input during validation or prediction phases.

        Args:
            input_images (list): A list of file paths to input images that are either pre-processed or need resizing.
            original_images (list or None): A list of file paths to original images (before any processing).
                                            If None, the input images are used as the original images.
            size (int, optional): The target size to which the input images should be resized. Defaults to 128.

    """

    def __init__(self, input_images, original_images, mask=None, size=128):
        self.input_images = input_images
        self.original_images = original_images
        self.size = size
        self.to_tensor = transforms.PILToTensor()
        self.transform_validation = CustomNormalize('validation')
        self.transform_validation_crop = CustomNormalize('validation_cropped')
        self.resize = transforms.Resize((size, size))
        self.mask = mask
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, item):
        if self.original_images:
            original_image = self.to_tensor(Image.open(self.original_images[item])) / 255
            resized_image = self.transform_validation_crop(
                self.resize(self.to_tensor(Image.open(self.input_images[item])) / 255))
            coordinates = self.input_images[item]
        else:
            original_image = self.to_tensor(Image.open(self.input_images[item])) / 255
            resized_image = self.transform_validation(self.resize(original_image))
            coordinates = self.input_images[item]
        if self.mask:
            mask = self.to_tensor(Image.open(self.mask[item]))
            mask[mask == 0] = 2
            mask[mask == 128] = 1
            mask[mask == 255] = 0
            return original_image, resized_image, mask

        return original_image, resized_image, coordinates








class CustomNormalize(object):
    def __init__(self, dataset_type):
        # Define means and stds for each dataset type
        self.dataset_stats = {
            "validation": {
                "mean": [0.5507591776382487, 0.36368202300487135, 0.28958523739594133],
                "std": [0.17806627200735062, 0.14194286672976278, 0.10991587430013793]
            },
            "validation_cropped": {
                "mean": [0.7267119568330208, 0.5110323962090456, 0.3950133781658404],
                "std": [0.1697782136782029, 0.16959714353640828, 0.13604012754018177]
            },
            "train_cropped": {
                "mean": [0.4533032407631631, 0.28634207832016756, 0.17689735851538596],
                "std": [0.22488837595254818, 0.20549813450645052, 0.1308072482768547]
            },
            "train": {
                "mean": [0.3384665271751848, 0.2027031821243444, 0.10234393640568978],
                "std": [0.15403509489004488, 0.11902604054121893, 0.05409505439008119]
            }
        }

        self.mean = torch.tensor(self.dataset_stats[dataset_type]["mean"]).view(-1, 1, 1)
        self.std = torch.tensor(self.dataset_stats[dataset_type]["std"]).view(-1, 1, 1)

    def __call__(self, image):
        # Normalize the entire image
        normalized_image = (image - self.mean) / self.std

        # Create a mask for non-black pixels (exactly zero across all channels)
        mask = (image != 0).any(dim=0, keepdim=True)

        # Apply the mask to keep black pixels unchanged
        normalized_image = normalized_image * mask + image * (~mask)

        return normalized_image

