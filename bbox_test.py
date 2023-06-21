import torchvision
torchvision.disable_beta_transforms_warning()
import torch
from torch import nn
from bbox_model import bbox_model
from  my_dataset import MyDataset
from torch.utils.data import DataLoader, Dataset
from paths import get_preprocessed_images_paths
from tqdm import tqdm
from torchvision.ops import masks_to_boxes
from PIL import Image

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = bbox_model(3)
model.load_state_dict(torch.load('best_model_1.pt'))
model.to(device)



model.eval()
val_loss = 0.0
delta_max = 0
dist_mean = []
with torch.no_grad():
    model.eval()
    train_images, train_masks, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths()
    test_dataset = MyDataset(val_images, val_masks)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for idx, (images, masks) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)

        bboxes = masks_to_boxes(masks)
        dist_mean.append(torch.mean(torch.abs(outputs - bboxes)))
        bboxes.to(device)
        image = images[0].detach().cpu() * 255
        image = image.to(torch.uint8)

        if (a := torch.max(torch.abs(outputs - bboxes))) > delta_max:
            delta_max = a



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
        if idx % 50 == 0:
            print(delta_max)
            print(sum(dist_mean)/(len(dist_mean)))
            img.save(f'pictures_test/{idx}.jpg')