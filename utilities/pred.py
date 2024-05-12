import numpy as np
import torch
import torchvision
import os
from glob import glob
from models import UNet_segmentation
from datasets import SegDatasetFromImages
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.ndimage import label, find_objects
from matplotlib import pyplot as plt
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('../config.yaml', 'r') as file:
    file = yaml.safe_load(file)
    config = file['wandb_config_seq']

images_path = r'C:\my files\REFUGE'
val_images_path = 'REFUGE-Validation400/**/*.jpg'
val_masks_path = 'REFUGE-Validation400-GT/**/*.bmp'
test_images_path = 'REFUGE-Test400/**/*.jpg'
test_masks_path = 'REFUGE-Test-GT/**/*.bmp'

val_images = sorted(glob(os.path.join(images_path, val_images_path), recursive=True), key=lambda x: os.path.basename(x))
val_masks = sorted(glob(os.path.join(images_path, val_masks_path), recursive=True), key=lambda x: os.path.basename(x))
test_images = sorted(glob(os.path.join(images_path, test_images_path), recursive=True),
                     key=lambda x: os.path.basename(x))
test_masks = sorted(glob(os.path.join(images_path, test_masks_path), recursive=True), key=lambda x: os.path.basename(x))

model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=config['base_dim'], depth=config['depth']).to(device)
model.load_state_dict(torch.load('../models/seg_best_model.pth'))

dataset = SegDatasetFromImages(val_images, val_masks)
dataloader = DataLoader(dataset)

resize = torchvision.transforms.Resize((256, 256))

for idx, (img_input, masks) in tqdm(enumerate(dataloader), total=len(dataloader)):
    original_img = img_input.clone()

    img_input = resize(img_input)
    masks = resize(masks)
    img_input = img_input.to(device)
    masks = masks.to(device)
    output = model(img_input)

    bbox_list = []

    output = output.detach().cpu().numpy()

    # Apply argmax and thresholding
    output = np.argmax(output[0], axis=0)
    output[output >= 1] = 1

    # Label connected components
    labeled_array, num_features = label(output, structure=[[1, 1, 1],
                                                           [1, 1, 1],
                                                           [1, 1, 1]])

    # Get unique labels and their counts
    labels, counts = np.unique(labeled_array, return_counts=True)

    # Remove background label (0)
    labels = labels[1:]
    counts = counts[1:]

    largest_patch_index = labels[np.argmax(counts)]
    largest_patch_indices = np.where(labeled_array == largest_patch_index)
    ratio_x, ratio_y = np.array(original_img.shape[2:]) / np.array(img_input.shape[2:])
    slice_x, slice_y = find_objects(labeled_array == largest_patch_index)[0]
    slice_x_start, slice_x_stop = int(slice_x.start * ratio_x), int(slice_x.stop * ratio_x)
    slice_y_start, slice_y_stop = int(slice_y.start * ratio_y), int(slice_y.stop * ratio_y)
    original_img_slice = original_img[0, 0, slice_x_start:slice_x_stop, slice_y_start:slice_y_stop]
    arr_ = np.squeeze(original_img_slice)
    plt.imshow(arr_)
    plt.show()
    bbox_list.append(bbox)



