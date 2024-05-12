import os

import torchvision
import torch
import wandb
import yaml

from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from utilities.models import UNet_segmentation
from torchmetrics import Dice
from losses_pytorch.focal_loss import FocalLoss
from losses_pytorch.dice_loss import GDiceLoss
from utilities.utils import plot_input_mask_output, get_preprocessed_images_paths
from utilities.datasets import SegDatasetFromTensors, SegDatasetFromImages
from multiprocessing import freeze_support

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login()

with open('config.yaml', 'r') as file:
    file = yaml.safe_load(file)
    config = file['wandb_config_seq']

train_images, train_masks, train_images_cropped_path, train_masks_cropped_path, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths(
    256, file_extension_img='.pt', file_extension_mask='.pt', refresh_search=False)

if __name__ == '__main__':
    freeze_support()

    with wandb.init(project='Unet-segmentation-pytorch', config=config):
        wandb.config.update(config)
        print(f'Wandb config: \n{wandb.config}')


        # Creating datasets and dataloaders for train, validation, and test

        val_dataset = SegDatasetFromTensors(input_images=val_images, label_images=val_masks,
                                           normalize_images=wandb.config.normalize_images)
        test_dataset = SegDatasetFromTensors(input_images=test_images, label_images=test_masks,
                                            normalize_images=wandb.config.normalize_images)

        # Creating dataloaders
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, prefetch_factor=2,
                                pin_memory_device='cuda', pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1, prefetch_factor=2,
                                 pin_memory_device='cuda', pin_memory=True)

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
        optimizer = optim.AdamW(model.parameters(), lr=wandb.config.lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True, threshold=0.0001)


        # Define number of epochs
        num_epochs = 30

        # Train the model
        patience = 6
        epochs_no_improve = 0
        best_val_loss = torch.inf

        torch.cuda.empty_cache()
        tolerance = 3

        for epoch in range(num_epochs):
            train_dataset = SegDatasetFromTensors(input_images=train_images[:],
                                                 label_images=train_masks[:],
                                                 cropped_input=train_images_cropped_path[:],
                                                 cropped_label=train_masks_cropped_path[:],
                                                 normalize_images=wandb.config.normalize_images,
                                                 ratio=0.6)
            train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True,num_workers=1, prefetch_factor=2, pin_memory_device='cuda', pin_memory=True)


            # Train
            model.train()
            train_loss = 0
            for idx, (img_input, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):

                #first pass
                img_input = img_input.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()

                outputs = model(img_input)
                loss_1 = criterion(outputs, masks.unsqueeze(1) if wandb.config.loss_type == 'dice' else masks)

                loss_1.backward()
                optimizer.step()
                train_loss += loss_1.item()

                if (idx % 10) == 0:
                    plot_input_mask_output(img_input=img_input[0], mask=masks[0], output=outputs[0], idx=idx, epoch=epoch)

            train_loss /= len(train_loader)
            print(f'Train loss: {train_loss}')
            wandb.log({'train_loss': train_loss})

            # Validation
            model.eval()  # Switch to evaluation mode
            val_loss = 0

            with torch.no_grad():
                for idx, (img_input, masks) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    img_input = img_input.to(device)
                    masks = masks.to(device)
                    outputs = model(img_input)
                    loss_1 = criterion(outputs, masks.unsqueeze(1) if wandb.config.loss_type == 'dice' else masks)
                    val_loss += loss_1.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss, epoch=epoch)
            print(f'Validation loss: {val_loss}')
            wandb.log({'val_loss': val_loss})

            # Early stopping
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'models/seg_best_model.pth')  # Save the best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping!")
                    break
