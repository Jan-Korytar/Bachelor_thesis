import os

import torchvision
import torch
from utilities.models import BboxModel
from utilities.datasets import BBoxDataset
from torch.utils.data import DataLoader
from utilities.utils import get_preprocessed_images_paths
from tqdm import tqdm
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('config.yaml', 'r') as file:
    file = yaml.safe_load(file)
    config = file['wandb_config_bbox_req']

size = 256
train_images, train_masks, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths(size=size)
model = BboxModel(in_channels=3,base_dim=config['base_dim'], depth=config['depth'], dropout=config['dropout'], batch_norm=config['batch_norm'],
                  img_dim=size).to(device)

model.load_state_dict(torch.load('models/bbox_best_model.pth'))
model.to(device)
model.eval()

val_loss = 0.0
delta_max = torch.tensor([0., 0., 0., 0.]).to(device)
dist_mean = []
with torch.no_grad():
    train_images, train_masks, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths()
    test_dataset = BBoxDataset(val_images, val_masks, normalize_images=config['normalize_images'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for idx, (images, bbox) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        bbox = bbox.to(device)

        outputs = model(images)
        outputs[:, 2:] += outputs[:, :2]
        bbox[:, 2:] += bbox[:, :2]


        dist_mean.append(torch.abs(outputs - bbox))

        if torch.any((a := torch.abs(outputs - bbox).squeeze()) > delta_max):
            print('max difference', (delta_max/size)*100)
            delta_max[delta_max<a] = a[delta_max<a].to(delta_max.dtype)

            print(f'Mean abs change: {sum(dist_mean) / (len(dist_mean))}')
