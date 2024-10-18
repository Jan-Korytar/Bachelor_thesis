import wandb
import yaml

from utilities.train import train
from utilities.utils import get_preprocessed_images_paths

# Fetch the preprocessed paths for images and masks
paths_dict = get_preprocessed_images_paths(128, file_extension_img='.pt', file_extension_mask='.pt',
                                           refresh_search=True)

train_images = paths_dict['train_images']
train_masks = paths_dict['train_masks']
train_images_cropped = paths_dict['train_images_cropped']
train_masks_cropped = paths_dict['train_masks_cropped']
val_images = paths_dict['val_images']
val_masks = paths_dict['val_masks']
val_images_cropped = paths_dict['val_images_cropped']
val_masks_cropped = paths_dict['val_masks_cropped']

# Load the sweep configuration from YAML file
with open('config.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)['wandb_sweep']


# Function to run the training for each sweep iteration
def sweep_train():
    # Pass the configuration to the training function
    train(wandb.config)  # wandb.config is automatically populated with current sweep parameters


if __name__ == '__main__':
    # Initialize sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="Unet-segmentation-pytorch")

    # Start the sweep, running as many iterations as you need
    wandb.agent(sweep_id, function=sweep_train)
