import gc
import os
import subprocess as sp

import numpy as np
import torch
import wandb
import yaml
from pytorch_toolbelt.losses import CrossEntropyFocalLoss, DiceLoss
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
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
    max_size_gb = 3.5
    model_size = get_model_size_in_gb(model)

    if model_size > max_size_gb:
        print(f"Model size ({model_size:.2f} GB) exceeds the limit of {max_size_gb} GB. Stopping the run.")
        del model
        wandb.finish()
        exit()
    print(f"Model size ({model_size:.2f} GB))")


def train(configuration):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths_dict = get_preprocessed_images_paths(size=configuration['size'], file_extension_img='.pt',
                                               file_extension_mask='.pt',
                                               refresh_search=True)
    train_images = paths_dict['train_images']
    train_masks = paths_dict['train_masks']
    train_images_cropped = paths_dict['train_images_cropped']
    train_masks_cropped = paths_dict['train_masks_cropped']
    val_images = paths_dict['val_images']
    val_masks = paths_dict['val_masks']
    val_images_cropped = paths_dict['val_images_cropped']
    val_masks_cropped = paths_dict['val_masks_cropped']

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wandb.login()
    with wandb.init(project='Unet-segmentation-pytorch', config=configuration, ):
        print(f'Wandb config: \n{wandb.config}')

        # Creating datasets and dataloaders for train, validation

        val_dataset = SegDatasetFromTensors(input_img=val_images, masks=val_masks[:],
                                            cropped_input=val_images_cropped[:],
                                            cropped_masks=val_masks_cropped[:],
                                            is_training=False, ratio=wandb.config.ratio,
                                            normalize_images=wandb.config.normalize_images)

        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=1, prefetch_factor=1, )

        model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=wandb.config.base_dim,
                                  depth=wandb.config.depth, growth_factor=wandb.config.growth_factor)

        get_model_size_in_gb(model)

        model.to(device)

        # Define loss function
        if wandb.config.loss_type == 'CCE':
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1., wandb.config.importance, wandb.config.importance])).to(
                device)
        elif wandb.config.loss_type == 'focal':
            criterion = CrossEntropyFocalLoss().to(device)
        elif wandb.config.loss_type == 'dice':
            criterion = DiceLoss(mode='multiclass').to(device)
        else:
            raise ValueError('No criterion specified')

        cce_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1., wandb.config.importance, wandb.config.importance])).to(
            device)
        val_criterion = DiceLoss(mode='multiclass').to(device)

        optimizer = optim.AdamW(model.parameters(), lr=wandb.config.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=1)

        num_epochs = 30
        plot_ratio = int(wandb.config.training_dataset_size / (3 * wandb.config.batch_size))

        patience = 3
        epochs_no_improve = 0
        best_val_loss = torch.inf

        ratio = wandb.config.ratio
        train_dataset = SegDatasetFromTensors(input_img=train_images,
                                              masks=train_masks,
                                              cropped_input=train_images_cropped,
                                              cropped_masks=train_masks_cropped,
                                              normalize_images=wandb.config.normalize_images,
                                              is_training=True,
                                              ratio=ratio)

        train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, num_workers=1,
                                  prefetch_factor=1, pin_memory_device='cuda', pin_memory=True, drop_last=True,
                                  sampler=torch.utils.data.RandomSampler(train_dataset,
                                                                         num_samples=wandb.config.training_dataset_size))

        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0
            for idx, (img_input, masks) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                desc=f'Epoch: {epoch}/{num_epochs}'):

                # forward pass
                img_input = img_input.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()

                outputs = model(img_input)
                if epoch == 0:
                    loss_1 = cce_criterion(outputs, masks).mean()
                else:
                    if wandb.config.loss_type == 'dice':
                        loss_1 = criterion(outputs, masks) + 2 * cce_criterion(outputs, masks)
                    else:
                        loss_1 = criterion(outputs, masks)

                loss_1.backward()
                optimizer.step()
                train_loss += loss_1.item()

                if idx + epoch == 0:
                    command = "nvidia-smi --query-gpu=memory.used --format=csv"
                    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
                    memory_used = [int(x.split()[0]) for i, x in enumerate(memory_used_info)][0]
                    if memory_used >= 3850:
                        print(f'model too big: {memory_used}, exiting')
                        del model, train_loader, train_dataset
                        torch.cuda.empty_cache()
                        wandb.finish()  # Stop the current W&B run
                        exit()

                if (idx % plot_ratio) == 0:
                    plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx,
                                           epoch=epoch, folder='train')

            train_loss /= len(train_loader)

            # Validation
            val_loss = 0

            model.eval()
            with torch.no_grad():
                for idx, (img_input, masks) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    img_input = img_input.to(device)
                    masks = masks.to(device)
                    outputs = model(img_input)
                    loss_1 = val_criterion(outputs, masks)
                    val_loss += loss_1.item()
                    if (idx % 10) == 0:
                        plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx,
                                               epoch=epoch, folder='val')

            val_loss /= len(val_loader)
            scheduler.step()

            print(f'Validation loss: {val_loss}, Train loss: {train_loss}')
            wandb.log({'val_loss': val_loss, 'train_loss': train_loss, 'epoch': epoch})

            # Early stopping
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                print(f'saving the best model: {wandb.config.name}_{wandb.config.size}.pth')
                torch.save(model.state_dict(), os.path.join(os.path.dirname(script_dir),
                                                            f'models/{wandb.config.name}_{wandb.config.size}.pth'))

                with open(f'models/data.yaml', 'w') as outfile:
                    yaml.dump(configuration, outfile, default_flow_style=False)
                # Save the best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    wandb.log({'val_loss': best_val_loss, 'step': epoch})
                    print("Early stopping!")
                    del model
                    torch.cuda.empty_cache()
                    break

    print('finishing')


def train_sweep(configuration):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths_dict = get_preprocessed_images_paths(size=configuration['size'], file_extension_img='.pt',
                                               file_extension_mask='.pt',
                                               refresh_search=True)
    train_images = paths_dict['train_images']
    train_masks = paths_dict['train_masks']
    train_images_cropped = paths_dict['train_images_cropped']
    train_masks_cropped = paths_dict['train_masks_cropped']

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wandb.login()
    with wandb.init(project='Unet-segmentation-pytorch', config=configuration, ):
        print(f'Wandb config: \n{wandb.config}')

        model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=wandb.config.base_dim,
                                  depth=wandb.config.depth, growth_factor=wandb.config.growth_factor)

        get_model_size_in_gb(model)

        model.to(device)

        # Define loss function
        if wandb.config.loss_type == 'CCE':
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1., wandb.config.importance, wandb.config.importance])).to(
                device)
        elif wandb.config.loss_type == 'focal':
            criterion = CrossEntropyFocalLoss().to(device)
        elif wandb.config.loss_type == 'dice':
            criterion = DiceLoss(mode='multiclass').to(device)
        else:
            raise ValueError('No criterion specified')

        cce_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1., wandb.config.importance, wandb.config.importance])).to(
            device)
        val_criterion = DiceLoss(mode='multiclass').to(device)

        optimizer = optim.AdamW(model.parameters(), lr=wandb.config.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=1)

        num_epochs = 15
        plot_ratio = int(wandb.config.training_dataset_size / (5 * wandb.config.batch_size))

        patience = 3
        epochs_no_improve = 0
        best_val_loss = torch.inf

        ratio = wandb.config.ratio
        train_dataset = SegDatasetFromTensors(input_img=train_images,
                                              masks=train_masks,
                                              cropped_input=train_images_cropped,
                                              cropped_masks=train_masks_cropped,
                                              normalize_images=wandb.config.normalize_images,
                                              is_training=True,
                                              ratio=ratio)

        indices = torch.randperm(len(train_dataset))[:wandb.config.training_dataset_size]

        subset = Subset(train_dataset, indices)

        # Create DataLoader for the subset
        train_loader = DataLoader(subset, batch_size=wandb.config.batch_size, shuffle=True)

        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0
            for idx, (img_input, masks) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                desc=f'Epoch: {epoch}/{num_epochs}'):

                # forward pass
                img_input = img_input.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()

                outputs = model(img_input)
                if epoch == 0:
                    loss_1 = cce_criterion(outputs, masks).mean()
                else:
                    if wandb.config.loss_type == 'dice':
                        loss_1 = criterion(outputs, masks) + 2 * cce_criterion(outputs, masks)
                    else:
                        loss_1 = criterion(outputs, masks)

                loss_1.backward()
                optimizer.step()
                train_loss += loss_1.item()

                if (idx % plot_ratio) == -1:
                    plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx,
                                           epoch=epoch, folder='train')
                if idx + epoch == 0:
                    command = "nvidia-smi --query-gpu=memory.used --format=csv"
                    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
                    memory_used = [int(x.split()[0]) for i, x in enumerate(memory_used_info)][0]
                    if memory_used >= 3850:
                        print(f'model too big: {memory_used}, exiting')
                        del model, train_loader, train_dataset
                        torch.cuda.empty_cache()
                        wandb.finish()  # Stop the current W&B run
                        exit()

            train_loss /= len(train_loader)

            # Validation
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for idx, (img_input, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):
                    img_input = img_input.to(device)
                    masks = masks.to(device)
                    outputs = model(img_input)
                    loss_1 = val_criterion(outputs, masks)
                    val_loss += loss_1.item()
                    if (idx % 10) == -1:
                        plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx,
                                               epoch=epoch, folder='val')

            val_loss /= len(train_loader)
            scheduler.step()

            print(f'Validation loss: {val_loss}, Train loss: {train_loss}')
            wandb.log({'val_loss': val_loss, 'train_loss': train_loss, 'epoch': epoch})

            # Early stopping
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss

            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    wandb.log({'val_loss': best_val_loss, 'step': epoch})
                    print("Early stopping!")
                    del model, train_loader, train_dataset
                    torch.cuda.empty_cache()
                    break

    print('Finishing run')
    del model, train_loader, train_dataset
    torch.cuda.empty_cache()
