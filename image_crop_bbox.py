import numpy as np
import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms
from paths import get_preprocessed_images_paths
from tqdm import tqdm
from torchvision.ops import masks_to_boxes
from PIL import Image
from torchvision import datapoints
from torchvision.io import read_image
from glob import glob
import os
from bbox_model import bbox_model
from multiprocessing import freeze_support
import multiprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'batch_size': 1,
    'lr': 1.781623386838983e-06,
    'base_dim': 64,
    'dropout': 0.3,
    'batch_norm': True,
    'loss_type': 'complete_iou',
    'decay': 0.7283639108891397,
    'normalize_images': False

}

model = bbox_model(in_channels=3, base_dim=config['base_dim'], dropout=config['dropout'],
                   batch_norm=config['batch_norm'])
model.load_state_dict(torch.load('best_model_3.pt'))
model.to(device)

images_path = r'C:\my files\REFUGE'
train_images_path = 'Training400/**/*.jpg'
train_masks_path = 'Annotation-Training400/Disc_Cup_Masks/**/*.bmp'
val_images_path = 'REFUGE-Validation400/**/*.jpg'
val_masks_path = 'REFUGE-Validation400-GT/**/*.bmp'
test_images_path = 'REFUGE-Test400/**/*.jpg'
test_masks_path = 'REFUGE-Test-GT/**/*.bmp'

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
    train_img = datapoints.Image(read_image(train_img))
    train_mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToTensor()(Image.open(train_mask))))
    val_img = datapoints.Image(read_image(val_img))
    val_mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToTensor()(Image.open(val_mask))))
    test_img = datapoints.Image(read_image(test_img))
    test_mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToTensor()(Image.open(test_mask))))

    train_img_resized = resize(train_img).to(device).unsqueeze(dim=0)
    val_img_resized = resize(val_img).to(device).unsqueeze(dim=0)
    test_img_resized = resize(test_img).to(device).unsqueeze(dim=0)

    train_bbox = model(train_img_resized.to(torch.float32)) + offset
    train_bbox[:, [0, 2]] *= train_img.shape[1]
    train_bbox[:, [0, 1]] = torch.clamp(train_bbox[:, [0, 1]], min=0)
    train_bbox[:, [1, 3]] *= train_img.shape[2]
    train_bbox[:, [2, 3]] = torch.clamp(train_bbox[:, [2, 3]], max=train_img.shape[1])
    train_bbox_true = masks_to_boxes(train_mask) + torch.tensor([-100, -100, 100, 100])
    train_bbox_true[:, [0, 1]] = torch.clamp(train_bbox_true[:, [0, 1]], min=0)
    train_bbox_true[:, [2, 3]] = torch.clamp(train_bbox_true[:, [2, 3]], max=train_img.shape[1])

    train_image_crop_model = train_img[:, int(train_bbox[:, 1]):int(train_bbox[:, 3]),
                             int(train_bbox[:, 0]):int(train_bbox[:, 2])]
    train_image_crop_true = train_img[:, int(train_bbox_true[:, 1]):int(train_bbox_true[:, 3]),
                            int(train_bbox_true[:, 0]):int(train_bbox_true[:, 2])]
    train_mask_crop_model = train_mask[:, int(train_bbox[:, 1]):int(train_bbox[:, 3]),
                            int(train_bbox[:, 0]):int(train_bbox[:, 2])]
    train_mask_crop_true = train_mask[:, int(train_bbox_true[:, 1]):int(train_bbox_true[:, 3]),
                           int(train_bbox_true[:, 0]):int(train_bbox_true[:, 2])]

    train_image_crop_model = to_image(train_image_crop_model)
    train_image_crop_true = to_image(train_image_crop_true)
    train_mask_crop_model = to_image_mask(train_mask_crop_model)
    train_mask_crop_true = to_image_mask(train_mask_crop_true)

    train_image_path_model = os.path.join(images_path, fr'cropped\train\model')
    train_image_path_true = os.path.join(images_path, fr'cropped\train\true')

    os.makedirs(train_image_path_model, exist_ok=True)
    os.makedirs(train_image_path_true, exist_ok=True)
    train_image_crop_model.save(os.path.join(train_image_path_model, fr'{idx}.jpg'))
    train_image_crop_true.save(os.path.join(train_image_path_true, fr'{idx}.jpg'))
    train_mask_crop_model.save(os.path.join(train_image_path_model, fr'{idx}.bmp'))
    train_mask_crop_true.save(os.path.join(train_image_path_true, fr'{idx}.bmp'))

    val_bbox = model(val_img_resized.to(torch.float32)) + offset
    val_bbox[:, [0, 2]] *= val_img.shape[1]
    val_bbox[:, [0, 1]] = torch.clamp(val_bbox[:, [0, 1]], min=0)
    val_bbox[:, [1, 3]] *= val_img.shape[2]
    val_bbox[:, [2, 3]] = torch.clamp(val_bbox[:, [2, 3]], max=val_img.shape[1])
    val_bbox_true = masks_to_boxes(val_mask) + torch.tensor([-100, -100, 100, 100])
    val_bbox_true[:, [0, 1]] = torch.clamp(val_bbox_true[:, [0, 1]], min=0)
    val_bbox_true[:, [2, 3]] = torch.clamp(val_bbox_true[:, [2, 3]], max=val_img.shape[1])

    val_image_crop_model = val_img[:, int(val_bbox[:, 1]):int(val_bbox[:, 3]),
                           int(val_bbox[:, 0]):int(val_bbox[:, 2])]
    val_image_crop_true = val_img[:, int(val_bbox_true[:, 1]):int(val_bbox_true[:, 3]),
                          int(val_bbox_true[:, 0]):int(val_bbox_true[:, 2])]
    val_mask_crop_model = val_mask[:, int(val_bbox[:, 1]):int(val_bbox[:, 3]),
                          int(val_bbox[:, 0]):int(val_bbox[:, 2])]
    val_mask_crop_true = val_mask[:, int(val_bbox_true[:, 1]):int(val_bbox_true[:, 3]),
                         int(val_bbox_true[:, 0]):int(val_bbox_true[:, 2])]

    val_image_crop_model = to_image(val_image_crop_model)
    val_image_crop_true = to_image(val_image_crop_true)
    val_mask_crop_model = to_image_mask(val_mask_crop_model)
    val_mask_crop_true = to_image_mask(val_mask_crop_true)

    val_image_path_model = os.path.join(images_path, fr'cropped\val\model')
    val_image_path_true = os.path.join(images_path, fr'cropped\val\true')

    os.makedirs(val_image_path_model, exist_ok=True)
    os.makedirs(val_image_path_true, exist_ok=True)
    val_image_crop_model.save(os.path.join(val_image_path_model, fr'{idx}.jpg'))
    val_image_crop_true.save(os.path.join(val_image_path_true, fr'{idx}.jpg'))
    val_mask_crop_model.save(os.path.join(val_image_path_model, fr'{idx}.bmp'))
    val_mask_crop_true.save(os.path.join(val_image_path_true, fr'{idx}.bmp'))

    test_bbox = model(test_img_resized.to(torch.float32)) + offset
    test_bbox[:, [0, 2]] *= test_img.shape[1]
    test_bbox[:, [0, 1]] = torch.clamp(test_bbox[:, [0, 1]], min=0)
    test_bbox[:, [1, 3]] *= test_img.shape[2]
    test_bbox[:, [2, 3]] = torch.clamp(test_bbox[:, [2, 3]], max=test_img.shape[1])
    test_bbox_true = masks_to_boxes(test_mask) + torch.tensor([-100, -100, 100, 100])
    test_bbox_true[:, [0, 1]] = torch.clamp(test_bbox_true[:, [0, 1]], min=0)
    test_bbox_true[:, [2, 3]] = torch.clamp(test_bbox_true[:, [2, 3]], max=test_img.shape[1])

    test_image_crop_model = test_img[:, int(test_bbox[:, 1]):int(test_bbox[:, 3]),
                            int(test_bbox[:, 0]):int(test_bbox[:, 2])]
    test_image_crop_true = test_img[:, int(test_bbox_true[:, 1]):int(test_bbox_true[:, 3]),
                           int(test_bbox_true[:, 0]):int(test_bbox_true[:, 2])]
    test_mask_crop_model = test_mask[:, int(test_bbox[:, 1]):int(test_bbox[:, 3]),
                           int(test_bbox[:, 0]):int(test_bbox[:, 2])]
    test_mask_crop_true = test_mask[:, int(test_bbox_true[:, 1]):int(test_bbox_true[:, 3]),
                          int(test_bbox_true[:, 0]):int(test_bbox_true[:, 2])]

    test_image_crop_model = to_image(test_image_crop_model)
    test_image_crop_true = to_image(test_image_crop_true)
    test_mask_crop_model = to_image_mask(test_mask_crop_model)
    test_mask_crop_true = to_image_mask(test_mask_crop_true)

    test_image_path_model = os.path.join(images_path, fr'cropped\test\model')
    test_image_path_true = os.path.join(images_path, fr'cropped\test\true')

    os.makedirs(test_image_path_model, exist_ok=True)
    os.makedirs(test_image_path_true, exist_ok=True)
    test_image_crop_model.save(os.path.join(test_image_path_model, fr'{idx}.jpg'))
    test_image_crop_true.save(os.path.join(test_image_path_true, fr'{idx}.jpg'))
    test_mask_crop_model.save(os.path.join(test_image_path_model, fr'{idx}.bmp'))
    test_mask_crop_true.save(os.path.join(test_image_path_true, fr'{idx}.bmp'))


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
