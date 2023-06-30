import os

import torchvision
torchvision.disable_beta_transforms_warning()
import torch
from bbox_model import bbox_model
from  my_dataset import BboxDataset
from torch.utils.data import DataLoader, Dataset
from paths import get_preprocessed_images_paths
from tqdm import tqdm
from torchvision.ops import masks_to_boxes
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
        'batch_size': 1,
        'lr': 1.781623386838983e-06,
        'base_dim': 60,
        'dropout': 0.3,
        'batch_norm': True,
        'loss_type': 'distance_iou',
        'decay': .728363910889139,
        'normalize_images': False

    }
model = bbox_model(in_channels=3, base_dim=config['base_dim'], dropout=config['dropout'], batch_norm=config['batch_norm'])
model.load_state_dict(torch.load('best_model_4.pt'))
model.to(device)



model.eval()
val_loss = 0.0
delta_max = torch.tensor([0., 0., 0., 0.]).to(device)
dist_mean = []
with torch.no_grad():
    model.train()
    train_images, train_masks, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths()
    test_dataset = BboxDataset(val_images, val_masks, normalize_images=config['normalize_images'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for idx, (images, masks) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images) * 126

        bboxes = masks_to_boxes(masks)
        dist_mean.append(torch.mean(torch.abs(outputs - bboxes)))
        bboxes.to(device)
        image = images[0].detach().cpu() * 255
        image = image.to(torch.uint8)

        if torch.any((a := torch.abs(outputs - bboxes).squeeze()) > delta_max):
            print('max difference',(delta_max/126)*100)
            delta_max[delta_max<a] = a[delta_max<a].to(delta_max.dtype)



            try:
                img = torchvision.transforms.ToPILImage()(torchvision.utils.draw_bounding_boxes(image,
                                                                                                torch.stack((
                                                                                                    torch.tensor(
                                                                                                        outputs[
                                                                                                            0],
                                                                                                        dtype=torch.uint8),
                                                                                                    torch.tensor(
                                                                                                        bboxes[
                                                                                                            0],
                                                                                                        dtype=torch.uint8)))))
            except ValueError:
                img = torchvision.transforms.ToPILImage()(image)
            print(sum(dist_mean) / (len(dist_mean)))
            os.makedirs('pictures_test', exist_ok=True)
            img.save(f'pictures_test/{idx}.jpg')
        if idx % 50 == 0:
            print(sum(dist_mean)/(len(dist_mean)))