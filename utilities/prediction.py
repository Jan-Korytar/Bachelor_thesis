import os
import shutil
from glob import glob

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, find_objects
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

from utilities.datasets import PredictionDataset, SegDatasetFromTensors
from utilities.models import UNet_segmentation
from utilities.train import get_model_size_in_gb
from utilities.utils import plot_input_mask_output, get_preprocessed_images_paths, plot_output_outputsmooth_bbox

with open('../models/data_segment.yaml', 'r') as file:
    config_seg = yaml.safe_load(file)
with open('../config.yaml', 'r') as file:
    file = yaml.safe_load(file)
    config_crop = file['config_cropping_model']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_images = sorted(glob(os.path.join(file['paths']['data_path'], r'REFUGE-Validation400\*.jpg')),
                     key=lambda x: os.path.basename(x))[:]
test_masks = sorted(glob(os.path.join(file['paths']['data_path'], r'REFUGE-Validation400-GT\Disc_Cup_Masks\*.bmp')),
                    key=lambda x: os.path.basename(x))[:]

test_dataset = PredictionDataset(test_images[:], original_images=None, mask=test_masks, size=128)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

paths = get_preprocessed_images_paths(128, file_extension_img='.pt', file_extension_mask='.pt', refresh_search=True)

val_images_cropped = paths['val_images_cropped']
val_masks_cropped = paths['val_masks_cropped']
val_images = paths['val_images']
val_masks = paths['val_masks']
val_dataset = SegDatasetFromTensors(input_img=val_images, masks=val_masks, cropped_input=val_images_cropped,
                                    cropped_masks=val_masks_cropped,
                                    is_training=False, ratio=0,
                                    normalize_images=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

original_img_shape = np.array((1634, 1634))
MARGIN_ERR = 0.20
EXPECTED_POSITION = np.array((61, 32))
EXPECTED_SHAPE = np.array((400, 400))


def process_images(loader, mode, size_1, size_2, localize=False):
    cropping_model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=config_crop['base_dim'],
                                       depth=config_crop['depth'], growth_factor=config_crop['growth_factor'])
    cropping_model.load_state_dict(torch.load(f'../models/cropping_model_{size_1}_final.pth', weights_only=False))

    print(get_model_size_in_gb(cropping_model))

    total_params = sum(p.numel() for p in cropping_model.parameters())
    trainable_params = sum(p.numel() for p in cropping_model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")

    segmentation_model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=config_seg['base_dim'],
                                           depth=config_seg['depth'], growth_factor=config_seg['growth_factor'])
    segmentation_model.load_state_dict(torch.load(f'../models/segmentation_model_{size_2}.pth', weights_only=False))

    print(get_model_size_in_gb(segmentation_model))

    total_params = sum(p.numel() for p in segmentation_model.parameters())
    trainable_params = sum(p.numel() for p in segmentation_model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}, {trainable_params}")

    if mode == 'two_models' or mode == 'two-models':

        if localize:
            cropping_model.to(device)
            cropping_model.eval()
            resize = transforms.Resize((size_1, size_1))
            shutil.rmtree('cropped_val')
            os.makedirs(f'cropped_val', exist_ok=True)
            sum_whole = 0
            sum_cropped = 0
            res = []
            for idx, (original_img, image, mask) in tqdm(enumerate(loader), total=len(loader)):
                original_img = original_img
                image = image.to(device)
                output = cropping_model(image)

                output = output[0]
                output_copy = output.cpu().detach().numpy()
                # plot_input_mask_output(original_img[0], mask[0, 0], output, idx, idx, 'pred_crop')
                output = torch.stack(
                    (output[0], torch.nn.functional.relu(output[1]) + torch.nn.functional.relu(output[2]))).argmax(
                    axis=0).detach().cpu().numpy()

                # Apply argmax and thresholding
                output[output >= 1] = 1
                smoothed_output = gaussian_filter(output.astype(float), sigma=3)
                smoothed_output[smoothed_output < 0.2] = 0
                smoothed_output[smoothed_output >= 0.2] = 1
                output = smoothed_output
                labeled_array, num_features = label(output, structure=[[1, 1, 1],
                                                                       [1, 1, 1],
                                                                       [1, 1, 1]])
                if np.sum(labeled_array) == 0:
                    labeled_array[EXPECTED_POSITION[0] - 3:EXPECTED_POSITION[0] + 3,
                    EXPECTED_POSITION[1] - 3:EXPECTED_POSITION[1] + 3] = 1

                labels, counts = np.unique(labeled_array, return_counts=True)
                best_idx = 1
                best_diff = np.inf
                for index in range(1, num_features + 1):
                    if index == 2:
                        pass
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
                difx = int((EXPECTED_SHAPE[0] - (slice_x_stop - slice_x_start)) / 2)
                x_min = 0
                y_min = 0
                x_max = original_img.shape[2]
                y_max = original_img.shape[3]
                slice_x_start = max(slice_x_start - difx, x_min)
                slice_x_stop = min(slice_x_stop + difx, x_max)

                slice_y_start, slice_y_stop = int(slice_y.start * ratio_y), int(slice_y.stop * ratio_y)
                dify = int((EXPECTED_SHAPE[1] - (slice_y_stop - slice_y_start)) / 2)
                slice_y_start = max(slice_y_start - dify, y_min)
                slice_y_stop = min(slice_y_stop + dify, y_max)

                if num_features >= 2:
                    plot_output_outputsmooth_bbox(output_copy, output, mask[0, 0].cpu().detach().numpy(),
                                                  (slice_x_start, slice_y_start,
                                                   slice_x_stop, slice_y_stop))

                original_img_slice = original_img[0, :, slice_x_start:slice_x_stop, slice_y_start:slice_y_stop]
                sum_whole += torch.sum(mask).item()
                sum_cropped += torch.sum(mask[0, :, slice_x_start:slice_x_stop, slice_y_start:slice_y_stop]).item()

                utils.save_image(original_img_slice,
                                 f'cropped_val/{idx}-{slice_x_start}_{slice_x_stop}_{slice_y_start}_{slice_y_stop}.png')
            print(np.mean(np.array(res), axis=0))
            print(sum_whole, sum_cropped, sum_cropped / sum_whole)
            del cropping_model


        segmentation_model.eval()
        segmentation_model.to(device)
        # sort by the index
        crop_images = sorted(glob(os.path.join('cropped_val/*png')),
                             key=lambda x: int(str(os.path.basename(x)).split('-')[0]))

        crop_images = PredictionDataset(crop_images, original_images=test_images, size=size_2)
        loader = torch.utils.data.DataLoader(crop_images, batch_size=1, shuffle=False)
        shutil.rmtree('submission_val/segmentation')
        os.makedirs(f'submission_val/segmentation', exist_ok=True)

        # for idx, (original_img, image, coordinates) in tqdm(enumerate(loader), total=len(loader)):
        #    image = image.to(device)
        #    output = segmentation_model(image)
        #    plot_input_mask_output(image.cpu().detach().numpy()[0], mask=np.zeros((256, 256)),
        #                           output=output.cpu().detach().numpy()[0], idx=idx,
        #                           epoch=idx, folder='pred')


        for idx, (original_img, image, coordinates) in tqdm(enumerate(loader), total=len(loader)):
            image = image.to(device)
            output = segmentation_model(image)

            coordinates = coordinates[0].split('.')[0].split('\\')[1].split('-')[1].split('_')
            coordinates = [int(cor) for cor in coordinates]

            resize = transforms.Resize((coordinates[1] - coordinates[0], coordinates[3] - coordinates[2]))

            if (idx + 1) % 100 == 0:
                plot_input_mask_output(image.cpu().detach().numpy()[0], mask=np.zeros((size_2, size_2)),
                                       output=output[0], idx=idx,
                                       epoch=idx, folder='pred')

            output = resize(output)
            output = output.cpu().detach().numpy()
            output = np.argmax(output[0], axis=0)
            output_copy = output.copy()
            output_copy[output_copy >= 1] = 1
            labeled_array, num_features = label(output_copy, structure=[[1, 1, 1],
                                                                        [1, 1, 1],
                                                                        [1, 1, 1]])

            labels, counts = np.unique(labeled_array, return_counts=True)
            best_idx = 1
            best_sum = 0
            for index in range(1, num_features + 1):
                sums = np.sum(np.where(labeled_array == index))
                if sums > best_sum:
                    best_sum = sums
                    best_idx = index

            output_safe = np.zeros_like(output)
            output_safe[labeled_array == best_idx] = output[labeled_array == best_idx]

            output = output_safe.astype(float)

            smoothed_output = gaussian_filter(output.astype(float), sigma=20)
            smoothed_output[smoothed_output < 0.6] = 0
            smoothed_output[(smoothed_output >= 0.6) & (smoothed_output < 1.5)] = 1
            smoothed_output[smoothed_output >= 1.5] = 2

            output = smoothed_output

            output[output == 0] = 255
            output[output == 1] = 128
            output[output == 2] = 0

            mask = np.ones_like(original_img[0, 0], dtype=np.uint8) * 255
            mask[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]] = output

            mask = Image.fromarray(mask)
            mask.save(f'submission_val/segmentation/T{idx + 1:04}.bmp')


process_images(test_loader, mode='two_models', size_1=128, size_2=128, localize=False)
