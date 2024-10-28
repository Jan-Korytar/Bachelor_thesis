import shutil
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_input_mask_output(img_input, mask, output, idx, epoch, folder='seg'):
    """
    Plot input, mask, and output images side by side for visualization during training.

    Parameters:
    - img_input (torch.Tensor): Input image tensor.
    - mask (torch.Tensor): Mask tensor.
    - output (torch.Tensor): Output tensor.
    - idx (int): Index for saving the plot.

    Returns:
    None
    """


    fig, ax = plt.subplots(ncols=3)

    # Process mask
    if type(mask) == torch.Tensor:
        mask = mask.detach().cpu().numpy()
    mask[mask == 1] = 126
    mask[mask == 2] = 255
    ax[2].imshow(mask, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('Mask')

    # Process output
    if type(output) == torch.Tensor:
        output = np.argmax(output.detach().cpu().numpy(), axis=0)
    pic = np.zeros_like(output)
    pic[output == 1] = 126
    pic[output == 2] = 255
    ax[1].imshow(pic, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title(f'Output image')

    # Process input
    if type(img_input) == torch.Tensor:
        image = img_input.detach().cpu().numpy()
    else:
        image = img_input
    image -= np.min(image)  # Subtract minimum value
    image /= (np.max(image) - np.min(image))  # Scale by range of values

    # Ensure image is within valid range
    image = np.clip(image, 0, 1)

    # Transpose to make it suitable for plotting (H, W, C)
    image = np.transpose(image, (1, 2, 0))

    # Plot the image
    ax[0].imshow(image, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Input Image')

    path = Path(__file__).resolve().parent
    # Display and save the plot
    if idx + epoch == 0:
        shutil.rmtree(path / f'pictures_training/{folder}', ignore_errors=True)
    os.makedirs(path / f'pictures_training/{folder}', exist_ok=True)
    fig.suptitle(f'Epoch: {epoch}, step: {idx}')
    plt.tight_layout()
    plt.savefig( path / f'pictures_training/{folder}/picture_{epoch}_{idx}.jpg')
    plt.close('all')


import os
import yaml
from glob import glob


def get_preprocessed_images_paths(size=128, file_extension_img='.jpg', file_extension_mask='.bmp', refresh_search=False,
                                  return_dict=True):
    """
    Retrieves or searches for preprocessed image and mask file paths, based on the specified parameters.
    It can refresh the search and store the results in a YAML file for future use or load previously saved paths.

    Parameters:
    ----------
    size : int, optional
        The size of the images/masks to be searched (default is 128).

    file_extension_img : str, optional
        The file extension for the image files (default is '.jpg').

    file_extension_mask : str, optional
        The file extension for the mask files (default is '.bmp').

    refresh_search : bool, optional
        If True, a new search is performed and the paths are saved to a YAML file.
        If False, previously saved paths are loaded from the YAML file (default is False).

    return_dict : bool, optional
        If True, returns the paths in a dictionary.
        If False, returns the paths as a tuple (default is True).

    Returns:
    -------
    dict or tuple
        If `return_dict` is True, a dictionary of lists with image and mask paths is returned.
        If `return_dict` is False, a tuple with lists of image and mask paths is returned.

    Notes:
    -----
    - The function looks for the configuration file `config.yaml` in the parent directory of the script's location.
    - The paths for preprocessed images and masks are saved and loaded from `utilities/preprocessed_paths.yaml`.
    - If the 'utilities' directory does not exist, it will be created automatically.

    Raises:
    -------
    FileNotFoundError
        If `config.yaml` or the expected preprocessed paths file is not found.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if refresh_search:
        # Load the config from the parent directory of the script
        config_path = os.path.join(script_dir, '../config.yaml')
        with open(config_path, 'r') as file:
            file = yaml.safe_load(file)
            paths = file['paths']
            images_path = paths['data_path']

        # Construct the image and mask paths relative to the current script's directory
        train_images_path = os.path.join(images_path, f'preprocessed/train/input/{size}/**/*{file_extension_img}')
        train_masks_path = os.path.join(images_path, f'preprocessed/train/labels/{size}/**/*{file_extension_mask}')
        train_images_cropped_path = os.path.join(images_path,
                                                 f'preprocessed/train/input_cropped/{size}/**/*{file_extension_img}')
        train_masks_cropped_path = os.path.join(images_path,
                                                f'preprocessed/train/labels_cropped/{size}/**/*{file_extension_mask}')
        train_images_resized_path = os.path.join(images_path, f'preprocessed/train_only_resize/input/{size}/**/*{file_extension_img}')
        train_masks_resized_path = os.path.join(images_path, f'preprocessed/train_only_resize/labels/{size}/**/*{file_extension_img}')
        val_images_path = os.path.join(images_path, f'preprocessed/validation/input/{size}/**/*{file_extension_img}')
        val_masks_path = os.path.join(images_path, f'preprocessed/validation/labels/{size}/**/*{file_extension_mask}')
        val_images_cropped_path = os.path.join(images_path,
                                               f'preprocessed/validation/input_cropped/{size}/**/*{file_extension_img}')
        val_masks_cropped_path = os.path.join(images_path,
                                              f'preprocessed/validation/labels_cropped/{size}/**/*{file_extension_mask}')
        test_images_path = os.path.join(images_path, f'preprocessed/test/input/{size}/**/*{file_extension_img}')
        test_masks_path = os.path.join(images_path, f'preprocessed/test/labels/{size}/**/*{file_extension_mask}')

        # Perform the file search
        train_images = sorted(glob(train_images_path, recursive=True), key=lambda x: os.path.basename(x))
        train_masks = sorted(glob(train_masks_path, recursive=True), key=lambda x: os.path.basename(x))
        train_images_cropped_path = sorted(glob(train_images_cropped_path, recursive=True), key=lambda x: os.path.basename(x))
        train_masks_cropped_path = sorted(glob(train_masks_cropped_path, recursive=True), key=lambda x: os.path.basename(x))
        val_images_cropped_path = sorted(glob(val_images_cropped_path, recursive=True),
                                         key=lambda x: os.path.basename(x))
        val_masks_cropped_path = sorted(glob(val_masks_cropped_path, recursive=True), key=lambda x: os.path.basename(x))
        train_images_resized_path = sorted(glob(train_images_resized_path, recursive=True),
                                           key=lambda x: os.path.basename(x))
        train_masks_resized_path = sorted(glob(train_masks_resized_path, recursive=True),
                                          key=lambda x: os.path.basename(x))
        val_images = sorted(glob(val_images_path, recursive=True), key=lambda x: os.path.basename(x))
        val_masks = sorted(glob(val_masks_path, recursive=True), key=lambda x: os.path.basename(x))
        test_images = sorted(glob(test_images_path, recursive=True), key=lambda x: os.path.basename(x))
        test_masks = sorted(glob(test_masks_path, recursive=True), key=lambda x: os.path.basename(x))

        # Prepare the dictionary with paths
        paths_dict = {
            'train_images': train_images,
            'train_masks': train_masks,
            'train_images_cropped': train_images_cropped_path,
            'train_masks_cropped': train_masks_cropped_path,
            'val_images': val_images,
            'val_masks': val_masks,
            'val_images_cropped': val_images_cropped_path,
            'val_masks_cropped': val_masks_cropped_path,
            'test_images': test_images,
            'test_masks': test_masks,
            'train_images_resized': train_images_resized_path,
            'train_masks_resized': train_masks_resized_path,
        }

        # Write the paths to the YAML file in the current script's directory
        yaml_path = os.path.join(script_dir, 'preprocessed_paths.yaml')
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(paths_dict, yaml_file)
    else:
        # Load the paths from the YAML file in the current script's directory
        yaml_path = os.path.join(os.path.dirname(script_dir), 'utilities/preprocessed_paths.yaml')
        with open(yaml_path, 'r') as yaml_file:
            paths_dict = yaml.safe_load(yaml_file)

        if return_dict:
            return paths_dict
        else:
            return (
                paths_dict['train_images'],
                paths_dict['train_masks'],
                paths_dict['train_images_cropped'],
                paths_dict['train_masks_cropped'],
                paths_dict['val_images'],
                paths_dict['val_masks'],
                paths_dict['test_images'],
                paths_dict['test_masks'],
                paths_dict['train_images_resized'],
                paths_dict['train_masks_resized'],
                paths_dict['val_images_cropped'],
                paths_dict['val_masks_cropped']
            )

    if return_dict:
        return paths_dict
    else:
        return (train_images, train_masks, train_images_cropped_path, train_masks_cropped_path, val_images,
                val_masks, test_images, test_masks, train_images_resized_path, train_masks_resized_path,
                val_images_cropped_path, val_masks_cropped_path)
