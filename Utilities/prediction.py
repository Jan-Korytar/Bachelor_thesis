import numpy as np
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from PIL import Image
from torchvision import datapoints
from torchvision.io import read_image
from glob import glob
import os
from Utilities.models import bbox_model, SegmentationModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_bbox = {
    'batch_size': 1,
    'lr': 1.781623386838983e-06,
    'base_dim': 64,
    'dropout': 0.3,
    'batch_norm': True,
    'loss_type': 'complete_iou',
    'decay': 0.7283639108891397,
    'normalize_images': False

}

config_seq = {
        'base_dim': 51,
        'batch_norm': True,
        'batch_size': 10,
        'importance': 1.9233588295561803,
        'loss_type': 'dice',
        'lr': 0.00021549015427425473,
        'mode': 'model',
        'normalize_images': False
    }

images_path = r'C:\my files\REFUGE'
val_images_path = 'REFUGE-Validation400/**/*.jpg'
val_masks_path = 'REFUGE-Validation400-GT/**/*.bmp'
test_images_path = 'REFUGE-Test400/**/*.jpg'
test_masks_path = 'REFUGE-Test-GT/**/*.bmp'

val_images = sorted(glob(os.path.join(images_path, val_images_path), recursive=True), key=lambda x: os.path.basename(x))
val_masks = sorted(glob(os.path.join(images_path, val_masks_path), recursive=True), key=lambda x: os.path.basename(x))
test_images = sorted(glob(os.path.join(images_path, test_images_path), recursive=True),
                     key=lambda x: os.path.basename(x))
test_masks = sorted(glob(os.path.join(images_path, test_masks_path), recursive=True), key=lambda x: os.path.basename(x))
model_bbox = bbox_model(in_channels=3, base_dim=config_bbox['base_dim'], dropout=config_bbox['dropout'],
                   batch_norm=config_bbox['batch_norm'])

model_bbox.load_state_dict(torch.load('../Models/best_model_3.pt'))
model_bbox.to(device)


seg_model = SegmentationModel(in_channels=3, out_channels=3, base_dim=config_seq['base_dim'], batch_norm=True).to(device)
seg_model.load_state_dict(torch.load('best_model_seq_1.pt'))

resize = transforms.Resize((126, 126), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
to_image_tensor = transforms.Compose(
    [transforms.ToImageTensor(), transforms.Resize((256, 256), antialias=True), transforms.ConvertImageDtype(torch.float32)])
to_image_mask = transforms.Compose([transforms.ToImageTensor(), transforms.Resize((256, 256), antialias=False,
                                                                                  interpolation=transforms.InterpolationMode.NEAREST_EXACT),
                                    transforms.ToPILImage()])

offset = (torch.tensor([-17.8452, -8.0020, +14.5419, +8.5310], dtype=torch.float32) / 100).to(device)
def process_images(val_img, val_mask, mode):
    val_img_str = val_img
    val_img = datapoints.Image(read_image(val_img))
    val_img_org = val_img.clone()
    val_mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToTensor()(Image.open(val_mask))))
    val_mask_org = val_mask.clone()
    val_img_resized = resize(val_img).to(device).unsqueeze(dim=0)

    val_bbox = model_bbox(val_img_resized.to(torch.float32)) + offset
    val_bbox[:, [0, 2]] *= val_img.shape[1]
    val_bbox[:, [0, 1]] = torch.clamp(val_bbox[:, [0, 1]], min=0)
    val_bbox[:, [1, 3]] *= val_img.shape[2]
    val_bbox[:, [2, 3]] = torch.clamp(val_bbox[:, [2, 3]], max=val_img.shape[1])

    val_image_crop_model = val_img[:, int(val_bbox[:, 1]):int(val_bbox[:, 3]),
                           int(val_bbox[:, 0]):int(val_bbox[:, 2])]
    val_mask_crop_model = val_mask[:, int(val_bbox[:, 1]):int(val_bbox[:, 3]),
                          int(val_bbox[:, 0]):int(val_bbox[:, 2])]

    size_cropped = val_image_crop_model.size()

    val_image_crop_model = datapoints.Image(to_image_tensor(val_image_crop_model))
    val_mask_crop_model = datapoints.Mask(to_image_mask(val_mask_crop_model))
    resize_crop = transforms.Compose([transforms.Resize(size_cropped[1:])])
    val_image_crop_model = datapoints.Image(seg_model(val_image_crop_model.to(device).unsqueeze(0)))
    val_image_crop_model = resize_crop(val_image_crop_model)

    output = np.argmax(val_image_crop_model[0].detach().cpu().numpy(), axis=0)
    pic = np.ones_like(output) * 255
    pic_original = np.ones_like(val_img_org[0].unsqueeze(0)) * 255
    pic[output == 1] = 128
    pic[output == 2] = 0

    pic_original[:, int(val_bbox[:, 1]):int(val_bbox[:, 3]),
                           int(val_bbox[:, 0]):int(val_bbox[:, 2])] = pic

    pic_original = Image.fromarray(pic_original.squeeze())

    val_image_path_model = os.path.join(images_path, fr'predictions\{mode}\segmentation')

    os.makedirs(val_image_path_model, exist_ok=True)
    pic_original.save(os.path.join(val_image_path_model, fr'{os.path.basename(val_img_str)[:5]}.bmp'))


for val_image, val_mask in tqdm(zip(val_images, val_masks)):
    process_images(val_image, val_mask, 'val')

for val_image, val_mask in tqdm(zip(test_images, test_masks)):
    process_images(val_image, val_mask, 'test')