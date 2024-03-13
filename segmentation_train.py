import os.path
import matplotlib.pyplot as plt
import numpy as np
import torchvision

torchvision.disable_beta_transforms_warning()
import torch
import torch.nn as nn
import wandb
import yaml
import torch.optim as optim
import sys

from glob import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities.models import UNet_segmentation
from torchmetrics import Dice

sys.path.insert(0, 'SegLoss-master/SegLoss-master/losses_pytorch')
from losses_pytorch.focal_loss import FocalLoss
from losses_pytorch.dice_loss import GDiceLoss
from utilities.utils import plot_input_mask_output, get_preprocessed_images_paths
from utilities.datasets import SegDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login()

with open('config.yaml', 'r') as file:
    file = yaml.safe_load(file)
    config = file['wandb_config_seq']

train_images, train_masks, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths(128)

with wandb.init(project='Unet-segmentation-pytorch', config=config, mode='disabled'):
    wandb.config.update(config)

    # Creating datasets and dataloaders for train, validation, and test
    train_dataset = SegDataset(train_images[:200], train_masks[:200], wandb.config.normalize_images)
    val_dataset = SegDataset(val_images, val_masks, wandb.config.normalize_images)
    test_dataset = SegDataset(test_images, test_masks, wandb.config.normalize_images)

    # Creating dataloaders
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)

    model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=wandb.config.base_dim,
                              depth=wandb.config.depth).to(device)

    wandb.watch(model, log_freq=20)

    # Define your loss function
    if wandb.config.loss_type == 'CCE':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., wandb.config.importance, wandb.config.importance])).to(
            device)
    elif wandb.config.loss_type == 'focal':
        criterion = FocalLoss().to(device)
    elif wandb.config.loss_type == 'dice':
        criterion = GDiceLoss().to(device)

    val_crit = Dice().to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)

    # Define number of epochs
    num_epochs = 20

    # Train the model
    patience = 3
    epochs_no_improve = 0
    best_val_loss = torch.inf

    torch.cuda.empty_cache()
    tolerance = 3

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for idx, (img_input, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):

            img_input = img_input.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            outputs = model(img_input)
            loss = criterion(outputs, masks.unsqueeze(1) if wandb.config.loss_type == 'dice' else masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (idx % 5) == 0:
                plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx,
                                       title=f'Epoch: {epoch}, step: {idx}')

        train_loss /= len(train_loader)
        print(train_loss)
        wandb.log({'train_loss': train_loss})

        # Validation
        model.eval()  # Switch to evaluation mode
        val_loss = 0

        with torch.no_grad():
            for idx, (img_input, masks) in tqdm(enumerate(val_loader), total=len(val_loader)):
                img_input = img_input.to(device)
                masks = masks.to(device)
                outputs = model(img_input)
                loss = criterion(outputs, masks.unsqueeze(1) if wandb.config.loss_type == 'dice' else masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(val_loss)
        wandb.log({'val_loss': val_loss})

        # Early stopping
        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/best_model.pth')  # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break
