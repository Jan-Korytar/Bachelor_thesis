import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from torchvision.ops import masks_to_boxes
from PIL import Image
from torchvision import datapoints
from torchvision.io import read_image
from glob import glob
import os
from models import BboxModel
from multiprocessing import freeze_support
import multiprocessing
import yaml
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('../config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

config = config_data['config']
model = BboxModel(in_channels=3, base_dim=config['base_dim'], dropout=config['dropout'],
                  batch_norm=config['batch_norm'])
model.load_state_dict(torch.load(config_data['model']['state_dict_path']))
model.to(device)

images_path = config_data['paths']['images_path']
train_images_path = config_data['paths']['train_images_path']
train_masks_path = config_data['paths']['train_masks_path']
val_images_path = config_data['paths']['val_images_path']
val_masks_path = config_data['paths']['val_masks_path']
test_images_path = config_data['paths']['test_images_path']
test_masks_path = config_data['paths']['test_masks_path']

train_images = sorted(glob(os.path.join(images_path, train_images_path), recursive=True),
                      key=lambda x: os.path.basename(x))
train_masks = sorted(glob(os.path.join(images_path, train_masks_path), recursive=True),
                     key=lambda x: os.path.basename(x))
val_images = sorted(glob(os.path.join(images_path, val_images_path), recursive=True), key=lambda x: os.path.basename(x))
val_masks = sorted(glob(os.path.join(images_path, val_masks_path), recursive=True), key=lambda x: os.path.basename(x))
test_images = sorted(glob(os.path.join(images_path, test_images_path), recursive=True),
                     key=lambda x: os.path.basename(x))
test_masks = sorted(glob(os.path.join(images_path, test_masks_path), recursive=True), key=lambda x: os.path.basename(x))

resize = transforms.Resize((126, 126), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
to_image = transforms.Compose(
    [transforms.ToImageTensor(), transforms.Resize((256, 256), antialias=True), transforms.ToPILImage()])
to_image_mask = transforms.Compose([transforms.ToImageTensor(), transforms.Resize((256, 256), antialias=False,
                                                                                  interpolation=transforms.InterpolationMode.NEAREST_EXACT),
                                    transforms.ToPILImage()])

offset = (torch.tensor([-17.8452, -8.0020, +14.5419, +8.5310], dtype=torch.float32) / 100).to(device)


def process_images(idx, train_img, train_mask, val_img, val_mask, test_img, test_mask):
    """
    Process and crop images and masks for training, validation, and testing.

    Args:
        idx (int): Index identifier for the images.
        train_img (str): File path or image data for training images.
        train_mask (str): File path or mask data for training masks.
        val_img (str): File path or image data for validation images.
        val_mask (str): File path or mask data for validation masks.
        test_img (str): File path or image data for testing images.
        test_mask (str): File path or mask data for testing masks.

    Returns:
        None: The function saves cropped images and masks in specified directories.
    """
    def process_data(img, mask, label):
        img_resized = resize(datapoints.Image(read_image(img)).to(device).unsqueeze(dim=0))
        bbox = model(img_resized.to(torch.float32)) + offset
        bbox[:, [0, 2]] *= img.shape[1]
        bbox[:, [0, 1]] = torch.clamp(bbox[:, [0, 1]], min=0)
        bbox[:, [1, 3]] *= img.shape[2]
        bbox[:, [2, 3]] = torch.clamp(bbox[:, [2, 3]], max=img.shape[1])
        bbox_true = masks_to_boxes(datapoints.Mask(transforms.RandomInvert(1)(transforms.ToTensor()(Image.open(mask)))))+torch.tensor([-100, -100, 100, 100])
        bbox_true[:, [0, 1]] = torch.clamp(bbox_true[:, [0, 1]], min=0)
        bbox_true[:, [2, 3]] = torch.clamp(bbox_true[:, [2, 3]], max=img.shape[1])

        image_crop_model = to_image(img[:, int(bbox[:, 1]):int(bbox[:, 3]), int(bbox[:, 0]):int(bbox[:, 2])])
        image_crop_true = to_image(img[:, int(bbox_true[:, 1]):int(bbox_true[:, 3]), int(bbox_true[:, 0]):int(bbox_true[:, 2])])
        mask_crop_model = to_image_mask(mask[:, int(bbox[:, 1]):int(bbox[:, 3]), int(bbox[:, 0]):int(bbox[:, 2])])
        mask_crop_true = to_image_mask(mask[:, int(bbox_true[:, 1]):int(bbox_true[:, 3]), int(bbox_true[:, 0]):int(bbox_true[:, 2])])

        image_path_model = os.path.join(images_path, fr'cropped\{label}\model')
        image_path_true = os.path.join(images_path, fr'cropped\{label}\true')

        os.makedirs(image_path_model, exist_ok=True)
        os.makedirs(image_path_true, exist_ok=True)
        image_crop_model.save(os.path.join(image_path_model, fr'{idx}.jpg'))
        image_crop_true.save(os.path.join(image_path_true, fr'{idx}.jpg'))
        mask_crop_model.save(os.path.join(image_path_model, fr'{idx}.bmp'))
        mask_crop_true.save(os.path.join(image_path_true, fr'{idx}.bmp'))

    process_data(train_img, train_mask, 'train')
    process_data(val_img, val_mask, 'val')
    process_data(test_img, test_mask, 'test')



def wrapper(args):
    idx, (train_img, train_mask, val_img, val_mask, test_img, test_mask) = args
    process_images(idx, train_img, train_mask, val_img, val_mask, test_img, test_mask)
    return


cpu = os.cpu_count() - 1

if __name__ == '__main__':
    freeze_support()

    with multiprocessing.Pool(processes=cpu) as pool:
        for _ in tqdm(pool.imap_unordered(wrapper, enumerate(
                zip(train_images, train_masks, val_images, val_masks, test_images, test_masks))),
                      total=len(test_masks)):
            pass
