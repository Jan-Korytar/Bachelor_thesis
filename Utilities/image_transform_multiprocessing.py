import os
from glob import glob
from multiprocessing import Pool, freeze_support
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from PIL import Image
from torchvision import datapoints
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
import numpy as yaml
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('config_paths.yaml', 'r') as file:
    paths_data = yaml.safe_load(file)

images_path = paths_data['paths']['images_path']
train_images_path = paths_data['paths']['train_images_path']
train_masks_path = paths_data['paths']['train_masks_path']
val_images_path = paths_data['paths']['val_images_path']
val_masks_path = paths_data['paths']['val_masks_path']
test_images_path = paths_data['paths']['test_images_path']
test_masks_path = paths_data['paths']['test_masks_path']

train_images = sorted(glob(os.path.join(images_path, train_images_path), recursive=True), key=lambda x: os.path.basename(x))
train_masks = sorted(glob(os.path.join(images_path, train_masks_path), recursive=True), key=lambda x: os.path.basename(x))
val_images = sorted(glob(os.path.join(images_path, val_images_path), recursive=True), key=lambda x: os.path.basename(x))
val_masks = sorted(glob(os.path.join(images_path, val_masks_path), recursive=True), key=lambda x: os.path.basename(x))
test_images = sorted(glob(os.path.join(images_path, test_images_path), recursive=True), key=lambda x: os.path.basename(x))
test_masks = sorted(glob(os.path.join(images_path, test_masks_path), recursive=True), key=lambda x: os.path.basename(x))


def image_train_transform(input_img, mask, alpha=10, size=256, mode='train'):
    if mode == 'train':

        input_img = datapoints.Image(read_image(input_img))
        mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToImageTensor()(Image.open(mask))))

        resize = transforms.Resize((size, size), antialias=True, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

        both_transforms = transforms.Compose([
            transforms.Resize((size, size), antialias=True, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.RandomPerspective(.1),
            transforms.RandomRotation(15)
        ])

        img_transforms = transforms.Compose([
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
        ])

        masks = []
        images = []
        masks.append(torch.tensor(resize(mask)))
        images.append(torch.tensor(resize(input_img)))
        for i in range(alpha-1):
            t_img, t_mask = both_transforms(input_img, mask)

            masks.append(torch.tensor(t_mask))
            t_img = img_transforms(t_img)
            images.append(torch.tensor(t_img))



        return images, masks


    else:

        input_img = datapoints.Image(read_image(input_img))
        mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToImageTensor()(Image.open(mask))))
        both_transforms = transforms.Compose([
            transforms.Resize((size, size), antialias=True, interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT)])
        input_img, mask = both_transforms(input_img, mask)
        return torch.unsqueeze(input_img, 0), torch.unsqueeze(mask, 0)


def process_image(i):
    mode = i[1]
    i = i[0]

    torchvision.disable_beta_transforms_warning()
    path = r'C:\my files\REFUGE\3'

    if not os.path.exists(os.path.join(path, fr'{mode}\images')):
        os.makedirs(os.path.join(path, fr'{mode}\images'), exist_ok=True)
        os.makedirs(os.path.join(path, fr'{mode}\labels'),  exist_ok=True)

    if mode == 'train':
        image, label = train_images[i], train_masks[i]
        #if os.path.exists(os.path.join(path, fr'training\masks\mask_{i}_9.bmp')):
        #    return

    elif mode == 'validation':
        image, label = val_images[i], val_masks[i]

        #if os.path.exists(os.path.join(path, fr'validation\masks\mask_{i}_0.bmp')):
        #    return

    elif mode == 'test':
        image, label = val_images[i], val_masks[i]
        #if os.path.exists(os.path.join(path, fr'\test\masks\mask_{i}_0.bmp')):
        #    return



    images, masks = image_train_transform(image, label, 8, 126, mode=mode)
    for j, (img, lab) in enumerate(zip(images, masks)):
        img = transforms.ToPILImage()(img)
        lab = transforms.ToPILImage()(lab)
        lab.save(os.path.join(path, fr'{mode}\images\mask_{i}_{j}.bmp'))
        img.save(os.path.join(path, fr'{mode}\labels\input_{i}_{j}.jpg'))


if __name__ == '__main__':
    freeze_support()

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'train'] for i in range(len(train_masks))]),
                      total=len(train_masks)):
            pass

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'test'] for i in range(len(val_masks))]),
                      total=len(val_masks)):
            pass

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'validation'] for i in range(len(val_masks))]),
                      total=len(val_masks)):
            pass

