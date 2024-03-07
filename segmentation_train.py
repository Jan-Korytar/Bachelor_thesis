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
    config = file['wandb_config']

train_images, train_masks, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths()

with wandb.init(project='Unet-segmentation-pytorch', config=config, mode="disabled"):
    wandb.config.update(config)

    train_dataset = SegDataset(train_images[:800], train_masks[:800], wandb.config.normalize_images)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)

    model = UNet_segmentation(in_channels=3, out_channels=3, base_dim=wandb.config.base_dim,
                              depth=wandb.config.depth).to(device)

    wandb.watch(model, log_freq=20)

    # Define your loss function
    if wandb.config.loss_type == 'CCE':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., wandb.config.importance, wandb.config.importance])).to(device)
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
    torch.cuda.empty_cache()
    best_val_loss = torch.inf
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

              if ((idx + 1) % 2) == 5:
                   plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx, title=f'Epoch: {epoch}, step: {idx}')

         train_loss /= len(train_loader)
         print(train_loss)
         wandb.log({'train_loss': train_loss})

         model.test()
         with torch.no_grad():
             for idx, (img_input, masks) in tqdm(enumerate(train_loader), total=len(test)):


