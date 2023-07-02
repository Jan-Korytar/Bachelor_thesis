import os.path

import matplotlib.pyplot as plt
import numpy as np
import torchmetrics
import torchvision
from glob import glob

torchvision.disable_beta_transforms_warning()
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import SegDatset
from paths import get_preprocessed_images_paths
from seg_model import SegmentationModel
import wandb
import torch.optim as optim

wandb.login()
import sys
from torchmetrics import Dice

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from focal_loss import FocalLoss
from dice_loss import GDiceLoss




def main():
    config = {
        'base_dim': 51,
        'batch_norm': True,
        'batch_size': 10,
        'importance': 1.9233588295561803,
        'loss_type': 'dice',
        'lr': 0.00021549015427425473,
        'mode': 'model',
        'normalize_images': False
    }

    with wandb.init(project='sweep_1', config=config):
        torch.cuda.empty_cache()
        config = wandb.config
        mode = config.mode
        path = 'C:\my files\REFUGE\cropped'
        train_images = sorted(glob(os.path.join(path, fr'train/{mode}/*jpg')))
        train_masks = sorted(glob(os.path.join(path, fr'train/{mode}/*bmp')))
        val_images = sorted(glob(os.path.join(path, fr'val/{mode}/*jpg')))
        val_masks = sorted(glob(os.path.join(path, fr'val/{mode}/*bmp')))

        train_dataset = SegDatset(train_images, train_masks, normalize_images=config.normalize_images)
        val_dataset = SegDatset(val_images, val_masks, normalize_images=config.normalize_images)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

        # Define your model
        model = SegmentationModel(in_channels=3, out_channels=3, base_dim=config.base_dim, batch_norm=config.batch_norm).to(device)

        wandb.watch(model, log_freq=20)

        # Define your loss function
        if config.loss_type == 'CCE':
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., config.importance, config.importance])).to(device)
        elif config.loss_type == 'focal':
            criterion = FocalLoss().to(device)
        elif config.loss_type == 'dice':
            criterion = GDiceLoss().to(device)
        val_crit = Dice().to(device)

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

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
            for idx, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1) if config.loss_type == 'dice' else masks)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if ((idx + 1) % 10) == -1:

                    fig, ax = plt.subplots(ncols=3)
                    output = np.argmax(outputs[0].detach().cpu().numpy(), axis=0)
                    mask = masks[0].detach().cpu().numpy()
                    mask[mask == 1] = 126
                    mask[mask == 2] = 255
                    ax[0].imshow(mask, cmap='gray')
                    ax[0].axis('off')
                    ax[0].set_title('mask')
                    pic = np.zeros_like(output)
                    pic[output == 1] = 126
                    pic[output == 2] = 255
                    ax[1].imshow(pic, cmap='gray')
                    ax[1].axis('off')
                    ax[1].set_title('output')
                    image = images[0].detach().cpu().numpy()
                    image = np.transpose(image, (1, 2, 0))
                    ax[2].imshow(image, cmap='gray')
                    ax[2].axis('off')
                    ax[2].set_title('input')
                    fig.suptitle('Training')
                    plt.savefig(f'pictures_training/picture_{idx}')
                    plt.show()

            train_loss /= len(train_loader)
            wandb.log({'train_loss': train_loss})
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx, (images, masks) in tqdm(enumerate(val_loader)):
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = val_crit(outputs, masks)
                    val_loss += (1 - loss.item())
                    if ((idx + 1) % 10) == -1:
                        tqdm.write(str(loss.item()))
                        fig, ax = plt.subplots(ncols=3)
                        output = np.argmax(outputs[0].detach().cpu().numpy(), axis=0)
                        mask = masks[0].detach().cpu().numpy()
                        mask[mask == 1] = 126
                        mask[mask == 2] = 255
                        ax[0].imshow(mask, cmap='gray')
                        ax[0].axis('off')
                        ax[0].set_title('mask')
                        pic = np.zeros_like(output)
                        pic[output == 1] = 126
                        pic[output == 2] = 255
                        ax[1].imshow(pic, cmap='gray')
                        ax[1].axis('off')
                        ax[1].set_title('output')
                        image = images[0].detach().cpu().numpy()
                        image = np.transpose(image, (1, 2, 0))
                        ax[2].imshow(image, cmap='gray')
                        ax[2].axis('off')
                        ax[2].set_title('input')
                        fig.suptitle('Validation')
                        os.makedirs('pictures_training', exist_ok=True)
                        plt.savefig(f'pictures_training/yay{idx}')
                        plt.show()

            val_loss /= len(val_loader)
            wandb.log({'val_loss': val_loss})
            if val_loss <= best_val_loss:
                tolerance = 4
                best_val_loss = val_loss
                tqdm.write(f'Saving the best model')
                torch.save(model.state_dict(), 'best_model_seq_1.pt')
            else:
                tolerance -= 1
                if tolerance <= 0:
                    wandb.log({'val_loss': best_val_loss})
                    break
            # Print progress
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


'''
sweep_config = {'method': 'random',
                'name': 'seq_sweep',
                'program': 'seq_train.py',
                'metric': {
                    'goal': 'minimize',
                    'name': 'val_loss'
                    },
                'parameters': {
        'mode':{'values': ['model']},
        'normalize_images':  {'values': [True, False]},
        'batch_norm':  {'values': [True, False]},
        'lr': {'min': 0.000001, 'max': 0.001},
        'base_dim': {'min': 16, 'max': 64},
        'batch_size': {'min': 1, 'max': 12},
        'importance': {'min': .5, 'max': 2.},
        'loss_type': {'values': ['focal', 'CCE', 'dice']}


    }
                }



sweep_id = wandb.sweep(sweep=sweep_config, project="sweep_1")
wandb.agent(sweep_id, function=main, count=45)
'''
main()
