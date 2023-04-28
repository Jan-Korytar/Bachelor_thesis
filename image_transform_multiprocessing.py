import os
from glob import glob
from multiprocessing import Pool, freeze_support

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datapoints
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

torchvision.disable_beta_transforms_warning()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

images_path = r'C:\my files\REFUGE'
train_images_path = 'Training400/**/*.jpg'
train_masks_path = 'Annotation-Training400/Disc_Cup_Masks/**/*.bmp'
val_images_path = 'REFUGE-Validation400/**/*.jpg'
val_masks_path = 'REFUGE-Validation400-GT/**/*.bmp'

train_images = sorted(glob(os.path.join(images_path, train_images_path), recursive=True))
train_masks = sorted(glob(os.path.join(images_path, train_masks_path), recursive=True))
val_images = sorted(glob(os.path.join(images_path, val_images_path), recursive=True))
val_masks = sorted(glob(os.path.join(images_path, val_masks_path), recursive=True))


def image_train_transform(input_img, mask, alpha=10, resize=256, mode='train'):
    if mode == 'train':

        input_img = datapoints.Image(read_image(input_img))
        mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToImageTensor()(Image.open(mask))))

        both_transforms = transforms.Compose([
            transforms.Resize(resize, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(.1),
            transforms.RandomRotation(90)
        ])

        img_transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        ])

        masks = []
        images = []
        for i in range(alpha):
            t_img, t_mask = both_transforms(input_img, mask)

            masks.append(t_mask)
            t_img = img_transforms(t_img)
            images.append(t_img)
        return images, masks
    else:

        input_img = datapoints.Image(read_image(input_img))
        mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToImageTensor()(Image.open(mask))))
        both_transforms = transforms.Compose([
            transforms.Resize(resize, antialias=True)])
        input_img, mask = both_transforms(input_img, mask)
        return torch.unsqueeze(input_img, 0), torch.unsqueeze(mask, 0)


def process_image(i):
    mode = i[1]
    i = i[0]

    torchvision.disable_beta_transforms_warning()
    if mode == 'train':
        image, label = train_images[i], train_masks[i]
        path = r'C:\my files\REFUGE\training'
        if os.path.exists(os.path.join(path, fr'masks\mask_{i}_9.bmp')):
            return
    elif mode == 'validation':
        image, label = val_images[i], val_masks[i]
        path = r'C:\my files\REFUGE\validation'

        if os.path.exists(os.path.join(path, fr'masks\mask_{i}_0.bmp')):
            return
    elif mode == 'test':
        pass

    images, labels = image_train_transform(image, label, 10, 256, mode=mode)
    for j, (img, lab) in enumerate(zip(images, labels)):
        img = transforms.ToPILImage()(img)
        lab = transforms.ToPILImage()(lab)
        img.save(os.path.join(path, fr'input\img_{i}_{j}.jpg'))
        lab.save(os.path.join(path, fr'masks\mask_{i}_{j}.bmp'))


transformed_path = r'C:\my files\REFUGE\training'
if __name__ == '__main__':

    torchvision.disable_beta_transforms_warning()
    freeze_support()

    with Pool(processes=os.cpu_count() - 1) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'train'] for i in range(len(train_masks))]),
                      total=len(train_masks)):
            pass
    transformed_path = r'C:\my files\REFUGE\validation'
    with Pool(processes=os.cpu_count() - 1) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'validation'] for i in range(len(val_masks))]),
                      total=len(val_masks)):
            pass
