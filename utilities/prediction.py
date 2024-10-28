import os
import shutil
from glob import glob

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import label, find_objects
from torchvision import transforms, utils
from tqdm import tqdm

from utilities.datasets import PredictionDataset
from utilities.models import UNet_segmentation
from utilities.train import get_model_size_in_gb
from utilities.utils import plot_input_mask_output

with open('../config.yaml', 'r') as file:
    file = yaml.safe_load(file)
    config_seg = file['config_cropping_model']
    config_crop = file['config_segmentation_model']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_images = sorted(glob(os.path.join(file['paths']['data_path'], r'REFUGE-Test400\Test400\*.jpg')),
                     key=lambda x: os.path.basename(x))[:]
test_masks = sorted(glob(os.path.join(file['paths']['data_path'], r'REFUGE-Test-GT\Disc_Cup_Masks\*.bmp')),
                    key=lambda x: os.path.basename(x))[:]

test_dataset = PredictionDataset(test_images[:], original_images=None, mask=test_masks, size=128)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

original_img_shape = np.array((1634, 1634))
MARGIN_ERR = 0.10
EXPECTED_POSITION = np.array((61, 32))


def process_images(loader, mode, size_1, size_2):
    cropping_model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=config_seg['base_dim'],
                                       depth=config_seg['depth'], growth_factor=config_seg['growth_factor'])
    cropping_model.load_state_dict(torch.load('../models/cropping_model.pth', weights_only=False))

    print(get_model_size_in_gb(cropping_model))

    segmentation_model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=config_crop['base_dim'],
                                           depth=config_crop['depth'], growth_factor=config_crop['growth_factor'])
    segmentation_model.load_state_dict(torch.load('../models/segmentation_model.pth', weights_only=False))

    if mode == 'two_models' or mode == 'two-models':
        cropping_model.to(device)
        cropping_model.train()
        resize = transforms.Resize((size_1, size_1))

        shutil.rmtree('cropped')
        os.makedirs(f'cropped', exist_ok=True)
        sum_whole = 0
        sum_cropped = 0
        res = []
        for idx, (original_img, image, mask) in tqdm(enumerate(loader), total=len(loader)):
            original_img = original_img
            image = image.to(device)
            output = cropping_model(image)

            output = output.detach().cpu().numpy()[0].argmax(axis=0)

            # Apply argmax and thresholding
            output[output >= 1] = 1
            labeled_array, num_features = label(output, structure=[[1, 1, 1],
                                                                   [1, 1, 1],
                                                                   [1, 1, 1]])

            labels, counts = np.unique(labeled_array, return_counts=True)
            best_idx = 1
            best_diff = np.inf
            for index in range(1, num_features + 1):
                if counts[index] < 6:
                    continue
                means = np.array(np.where(labeled_array == index)).mean(axis=1)
                diff = np.mean(np.abs(EXPECTED_POSITION - means))
                if diff < best_diff:
                    best_diff = diff
                    best_idx = index

            largest_patch_indices = np.where(labeled_array == best_idx)
            ratio_x, ratio_y = np.array(original_img.shape[2:]) / np.array(image.shape[2:])
            slice_x, slice_y = find_objects(labeled_array == best_idx)[0]

            slice_x_start, slice_x_stop = int(slice_x.start * ratio_x), int(slice_x.stop * ratio_x)
            difx = int((slice_x_stop - slice_x_start) * MARGIN_ERR)
            x_min = 0
            y_min = 0
            x_max = original_img.shape[2]
            y_max = original_img.shape[3]
            slice_x_start = max(slice_x_start - difx, x_min)
            slice_x_stop = min(slice_x_stop + difx, x_max)

            slice_y_start, slice_y_stop = int(slice_y.start * ratio_y), int(slice_y.stop * ratio_y)
            dify = int((slice_y_stop - slice_y_start) * MARGIN_ERR)
            slice_y_start = max(slice_y_start - dify, y_min)
            slice_y_stop = min(slice_y_stop + dify, y_max)

            res.append([slice_x_stop - slice_x_start, slice_y_stop - slice_y_start])

            original_img_slice = original_img[0, :, slice_x_start:slice_x_stop, slice_y_start:slice_y_stop]
            sum_whole += torch.sum(mask).item()
            sum_cropped += torch.sum(mask[0, :, slice_x_start:slice_x_stop, slice_y_start:slice_y_stop]).item()

            utils.save_image(original_img_slice,
                             f'cropped/{idx}-{slice_x_start}_{slice_x_stop}_{slice_y_start}_{slice_y_stop}.png')
        print(np.mean(np.array(res), axis=0))
        print(sum_whole, sum_cropped, sum_cropped / sum_whole)
        exit()
        del cropping_model
        segmentation_model.eval()
        segmentation_model.to(device)
        crop_images = sorted(glob(os.path.join('cropped/*png')), key=lambda x: os.path.basename(x))

        crop_images = PredictionDataset(crop_images, original_images=test_images, size=size_2)
        loader = torch.utils.data.DataLoader(crop_images, batch_size=1, shuffle=False)
        shutil.rmtree('submission/segmentation')
        os.makedirs(f'submission/segmentation', exist_ok=True)
        means = []
        for idx, (original_img, image, coordinates) in tqdm(enumerate(loader), total=len(loader)):
            image = image.to(device)
            output = segmentation_model(image)
            coordinates = coordinates[0].split('.')[0].split('\\')[1].split('-')[1].split('_')
            coordinates = [int(cor) for cor in coordinates]
            resize = transforms.Resize((coordinates[1] - coordinates[0], coordinates[3] - coordinates[2]))
            output = resize(output)
            output = output.cpu().detach().numpy()
            output = np.argmax(output[0], axis=0)
            plot_input_mask_output(image.cpu().detach().numpy()[0], mask=np.zeros((256, 256)), output=output, idx=idx,
                                   epoch=idx, folder='pred')
            output[output == 0] = 255
            output[output == 1] = 128
            output[output == 2] = 0
            mask = np.ones_like(original_img[0, 0], dtype=np.uint8) * 255
            mask[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]] = output

            mask = Image.fromarray(mask)
            mask.save(f'submission/segmentation/T{idx + 1:04}.bmp')


process_images(test_loader, mode='two_models', size_1=128, size_2=256)
