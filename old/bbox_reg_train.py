import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
import yaml
from torch.utils.data import DataLoader, RandomSampler
from torchvision.ops import complete_box_iou_loss, sigmoid_focal_loss
from tqdm import tqdm

from utilities.datasets import BBoxDataset
from utilities.models import BboxModel
from utilities.utils import get_preprocessed_images_paths

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login()

with open('../config.yaml', 'r') as file:
    file = yaml.safe_load(file)
    config = file['wandb_config_bbox_req']

size = 128
train_images, train_masks, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths(size=size)


def train(configuration=None):
    wandb.init(project='Unet-segmentation-pytorch', config=configuration)

    # Creating datasets and dataloaders for train, validation, and test
    train_dataset = BBoxDataset(train_images[:], train_masks[:], wandb.config.normalize_images)
    val_dataset = BBoxDataset(val_images, val_masks, wandb.config.normalize_images)
    test_dataset = BBoxDataset(test_images, test_masks, wandb.config.normalize_images)

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=100)  # Not ideal, replacement

    # Creating dataloaders
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, sampler=train_sampler)

    model = BboxModel(in_channels=3, base_dim=wandb.config.base_dim, depth=wandb.config.depth,
                      batch_norm=wandb.config.batch_norm, dropout=wandb.config.dropout, img_dim=size).to(device)

    wandb.watch(model, log_freq=20)

    criterion = nn.MSELoss().to(device)
    val_crit = nn.MSELoss().to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)

    # Define number of epochs
    num_epochs = 30

    # Train the model
    patience = 3
    epochs_no_improve = 0
    best_val_loss = torch.inf

    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for idx, (img_input, label_bbox) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                 desc=f'Epoch: {epoch}'):

            img_input = img_input.to(device)
            label_bbox = label_bbox.to(device)
            optimizer.zero_grad()

            output_train = model(img_input)

            if epoch < 1:
                loss = criterion(output_train, label_bbox)
            else:
                if wandb.config.loss_type == 'focal':
                    output_train[:, 2:] += output_train[:, :2]
                    label_bbox[:, 2:] += label_bbox[:, :2]
                    loss = sigmoid_focal_loss(output_train, label_bbox, reduction='mean')
                elif wandb.config.loss_type == 'iou':
                    output_train[:, 2:] += output_train[:, :2]
                    label_bbox[:, 2:] += label_bbox[:, :2]
                    loss = complete_box_iou_loss(output_train, label_bbox, reduction='mean')
                elif wandb.config.loss_type == 'MSE':
                    loss = criterion(output_train, label_bbox)
                else:
                    loss = criterion(output_train, label_bbox)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (idx % 20) == 0 and epoch > 2:
                try:
                    img_input = (img_input[0].detach().cpu() * 255).to(torch.uint8)  #
                    output = torch.squeeze(output_train[0].detach().cpu()).to(torch.uint8)
                    output[2:] += output[:2]
                    label = torch.squeeze(label_bbox[0].clone().cpu()).to(torch.uint8)
                    label[2:] += label[:2]
                    trans = torchvision.transforms.ToPILImage()
                    bboxes = torch.stack((output, label))
                    img = trans(
                        torchvision.utils.draw_bounding_boxes(img_input, bboxes, colors=['red', 'green'], width=2,
                                                              labels=['output', 'true']))
                    img.save(f'pictures_training/bbox/bbox_{epoch}_{idx}.jpg')
                except ValueError as f:
                    print(f)
                    pass

        train_loss /= len(train_loader)
        wandb.log({'train_loss': train_loss})

        # Validation
        model.eval()  # Switch to evaluation mode
        val_loss = 0

        with torch.no_grad():
            for idx, (img_input, label_bbox) in tqdm(enumerate(val_loader), total=len(val_loader)):
                img_input = img_input.to(device)
                label_bbox = label_bbox.to(device)
                output_train = model(img_input)
                loss = val_crit(output_train, label_bbox)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Val loss: {val_loss}, {output_train[0]}, {label_bbox[0]}')
        wandb.log({'val_loss': val_loss})

        # Early stopping
        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'../models/bbox_best_model.pth')  # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                wandb.log({'val_loss': best_val_loss})
                print("Early stopping!")
                print(config)
                break


train(config)
'''
sweep_config = {
    "name": "Sweep_1",
    "method": "random",  # You can choose grid or random
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32]},
        "base_dim": {"values": [32, 64, 80]},
        "lr": {'min':0.0001, 'max':0.01},
        'dropout': {'values': [0, 0.1, 0.2, 0.3, 0.4]},
        'batch_norm': {'values': [True, False]},
        'normalize_images': {'values': [True, False]},
        'depth':{'values': [4,5,6,7]},
        "loss_type": {"values": ['MSE', 'iou']}
    }
}



sweep_id = wandb.sweep(sweep_config, project='sweep')
wandb.agent(sweep_id, function=train)
'''
