import os

import torch
import wandb
from pytorch_toolbelt.losses import CrossEntropyFocalLoss, DiceLoss
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utilities.datasets import SegDatasetFromTensors
from utilities.models import UNet_segmentation
from utilities.utils import plot_input_mask_output, get_preprocessed_images_paths


def get_model_size_in_gb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    total_size_gb = total_size / (1024 ** 3)  # Convert to GB
    return total_size_gb


# Function to check if model exceeds 4GB and stop the run if true
def check_model_size_and_stop(model):
    max_size_gb = 4  # 4 GB limit
    model_size = get_model_size_in_gb(model)

    if model_size > max_size_gb:
        print(f"Model size ({model_size:.2f} GB) exceeds the limit of {max_size_gb} GB. Stopping the run.")
        wandb.finish()  # Stop the current W&B run
        exit()
        return True
    return False


def train(configuration):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths_dict = get_preprocessed_images_paths(128, file_extension_img='.pt', file_extension_mask='.pt',
                                               refresh_search=False)
    train_images = paths_dict['train_images']
    train_masks = paths_dict['train_masks']
    train_images_cropped = paths_dict['train_images_cropped']
    train_masks_cropped = paths_dict['train_masks_cropped']
    val_images = paths_dict['val_images']
    val_masks = paths_dict['val_masks']
    val_images_cropped = paths_dict['val_images_cropped']
    val_masks_cropped = paths_dict['val_masks_cropped']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    torch.cuda.empty_cache()
    wandb.login()
    with wandb.init(project='Unet-segmentation-pytorch', config=configuration):
        print(f'Wandb config: \n{wandb.config}')

        # Creating datasets and dataloaders for train, validation

        val_dataset = SegDatasetFromTensors(input_img=val_images, masks=val_masks, cropped_input=val_images_cropped,
                                            cropped_masks=val_masks_cropped,
                                            is_training=False, ratio=wandb.config.ratio,
                                            normalize_images=wandb.config.normalize_images)

        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=1, prefetch_factor=1)

        model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=wandb.config.base_dim,
                                  depth=wandb.config.depth, growth_factor=wandb.config.growth_factor).to(device)

        wandb.watch(model, log_freq=20)

        # Define loss function
        if wandb.config.loss_type == 'CCE':
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1., wandb.config.importance, wandb.config.importance])).to(
                device)
        elif wandb.config.loss_type == 'focal':
            criterion = CrossEntropyFocalLoss().to(device)
        elif wandb.config.loss_type == 'dice':
            criterion = DiceLoss(mode='multiclass').to(device)

        cce_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1., wandb.config.importance, wandb.config.importance])).to(
            device)

        optimizer = optim.AdamW(model.parameters(), lr=wandb.config.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=1)

        num_epochs = 20
        plot_ratio = int(wandb.config.training_dataset_size / (5 * wandb.config.batch_size))

        patience = 4
        epochs_no_improve = 0
        best_val_loss = torch.inf
        for epoch in range(num_epochs):
            ratio = wandb.config.ratio
            train_dataset = SegDatasetFromTensors(input_img=train_images,
                                                  masks=train_masks,
                                                  cropped_input=train_images_cropped,
                                                  cropped_masks=train_masks_cropped,
                                                  normalize_images=wandb.config.normalize_images,
                                                  is_training=True,
                                                  ratio=ratio)

            train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, num_workers=1,
                                      prefetch_factor=1, pin_memory_device='cuda', pin_memory=True,
                                      sampler=torch.utils.data.RandomSampler(train_dataset,
                                                                             num_samples=wandb.config.training_dataset_size))

            # Train
            model.train()
            train_loss = 0
            for idx, (img_input, masks) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                desc=f'Epoch: {epoch}/{num_epochs}'):

                # first pass
                img_input = img_input.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()

                outputs = model(img_input)
                if epoch == 0:
                    loss_1 = cce_criterion(outputs, masks)
                else:
                    loss_1 = criterion(outputs, masks)

                loss_1.backward()
                optimizer.step()
                train_loss += loss_1.item()

                if (idx % plot_ratio) == 0:
                    plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx,
                                           epoch=epoch, folder='train')

            train_loss /= len(train_loader)
            print(f'Train loss: {train_loss}')
            wandb.log({'train_loss': train_loss})

            # Validation
            val_loss = 0

            model.eval()
            with torch.no_grad():
                for idx, (img_input, masks) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    img_input = img_input.to(device)
                    masks = masks.to(device)
                    outputs = model(img_input)
                    loss_1 = cce_criterion(outputs, masks)
                    val_loss += loss_1.item()
                    if (idx % 10) == 0:
                        plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx,
                                               epoch=epoch, folder='val')

            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            print(f'Validation loss: {val_loss}')
            wandb.log({'val_loss': val_loss})

            # Early stopping
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(os.path.dirname(script_dir),
                                                            'models/segmentation_best_model_1.pth'))  # Save the best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping!")
                    break
