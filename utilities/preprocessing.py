import os
from glob import glob
from utils import get_preprocessed_images_paths
from multiprocessing import Pool, freeze_support
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from PIL import Image
from torchvision import tv_tensors as datapoints
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
import yaml
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('../config.yaml', 'r') as file:
    paths_data = yaml.safe_load(file)['paths']

images_path = paths_data['data_path']
train_images_path = paths_data['train_images_path']
train_masks_path = paths_data['train_masks_path']
val_images_path = paths_data['val_images_path']
val_masks_path = paths_data['val_masks_path']
test_images_path = paths_data['test_images_path']
test_masks_path = paths_data['test_masks_path']

train_images = sorted(glob(os.path.join(images_path, train_images_path + '/**/*.jpg'), recursive=True), key=lambda x: os.path.basename(x))
train_masks = sorted(glob(os.path.join(images_path, train_masks_path + '/**/*.bmp'), recursive=True), key=lambda x: os.path.basename(x))
val_images = sorted(glob(os.path.join(images_path, val_images_path)+ '/**/*.jpg', recursive=True), key=lambda x: os.path.basename(x))
val_masks = sorted(glob(os.path.join(images_path, val_masks_path + '/**/*.bmp'), recursive=True), key=lambda x: os.path.basename(x))
test_images = sorted(glob(os.path.join(images_path, test_images_path + '/**/*.jpg'), recursive=True), key=lambda x: os.path.basename(x))
test_masks = sorted(glob(os.path.join(images_path, test_masks_path + '/**/*.bmp'), recursive=True), key=lambda x: os.path.basename(x))



def image_preprocessing(image, mask, copies=10, size=128, mode='train'):
    image = datapoints.Image(read_image(image))
    mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToImage()(Image.open(mask))))
    crop_coordinates = torchvision.ops.masks_to_boxes(mask).to(int)[0]
    
    crop_coordinates_width_height = torch.tensor((crop_coordinates[1], crop_coordinates[0], crop_coordinates[3] - crop_coordinates[1] ,crop_coordinates[2] - crop_coordinates[0]))
    randomized = torch.randint(0, 256, (4,))
    crop_coordinates_width_height[2:] += randomized[2:]
    crop_coordinates_width_height[:2] -= randomized[:2]
    crop_coordinates_width_height[2:] += randomized[:2]
    mask_cropped = transforms.functional.crop(mask, *crop_coordinates_width_height)
    image_cropped = transforms.functional.crop(image, *crop_coordinates_width_height)

    resize = transforms.Resize((size, size), antialias=True,
                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    if mode == 'train':
        both_transforms = transforms.Compose([
            transforms.Resize((size, size), antialias=True, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomPerspective(.1),
            transforms.RandomRotation(15)
        ])

        img_transforms = transforms.Compose([
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.025)
        ])

        masks = []
        images = []
        masks_cropped = []
        images_cropped = []
        masks.append(resize(mask))
        images.append(resize(image))
        masks_cropped.append(resize(mask_cropped))
        images_cropped.append(resize(image_cropped))

        # cropping
        for i in range(copies - 1):
            t_img, t_mask, t_mask_cropped, t_image_cropped = both_transforms(image, mask, mask_cropped, image_cropped)

            masks.append(t_mask)
            masks_cropped.append(t_mask_cropped)
            t_img = img_transforms(t_img)
            t_image_cropped = img_transforms(t_image_cropped)
            images.append(t_img)
            images_cropped.append(t_image_cropped)



        return images, masks, images_cropped, masks_cropped

    else:

        image, mask = resize(image, mask)
        return torch.unsqueeze(image, 0), torch.unsqueeze(mask, 0)


#image_preprocessing(train_images[0], train_masks[0], 10, 128, mode='lol' )

def process_image(i):
    mode = i[1]
    i = i[0]
    for size in [128, 256, 512]:

        path = r'C:\my files\REFUGE\preprocessed'
        if not os.path.exists(os.path.join(path, fr'{mode}\input\{size}')) or not os.path.exists(os.path.join(path, fr'{mode}\input_cropped\{size}')):
            os.makedirs(os.path.join(path, fr'{mode}\input\{size}'), exist_ok=True)
            os.makedirs(os.path.join(path, fr'{mode}\labels\{size}'),  exist_ok=True)
            if mode == 'train':
                os.makedirs(os.path.join(path, fr'{mode}\input_cropped\{size}'), exist_ok=True)
                os.makedirs(os.path.join(path, fr'{mode}\labels_cropped\{size}'),  exist_ok=True)

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




        if mode == 'train':
            images, masks, images_cropped, masks_cropped = image_preprocessing(image, label, 5, size, mode=mode)
            for j, (image, mask, image_cropped, mask_cropped) in enumerate(zip(images, masks, images_cropped, masks_cropped)):


                to_PIL = transforms.functional.to_pil_image



                image = to_PIL(image)
                mask = to_PIL(mask)
                image_cropped = to_PIL(image_cropped)
                mask_cropped = to_PIL(mask_cropped)

                mask.save(os.path.join(path, fr'{mode}\labels\{size}\mask_{i}_{j}.bmp'))
                image.save(os.path.join(path, fr'{mode}\input\{size}\input_{i}_{j}.jpg'))
                mask_cropped.save(os.path.join(path, fr'{mode}\labels_cropped\{size}\mask_{i}_{j}.bmp'))
                image_cropped.save(os.path.join(path, fr'{mode}\input_cropped\{size}\input_{i}_{j}.jpg'))
        else:
            images, masks = image_preprocessing(image, label, 5, size, mode=mode)
            for j, (image, mask) in enumerate(
                    zip(images, masks)):
                to_PIL = transforms.functional.to_pil_image

                image = to_PIL(image)
                mask = to_PIL(mask)
                mask.save(os.path.join(path, fr'{mode}\labels\{size}\mask_{i}_{j}.bmp'))
                image.save(os.path.join(path, fr'{mode}\input\{size}\input_{i}_{j}.jpg'))


# process_image((0, 'validation'))


if __name__ == '__main__':
    freeze_support()

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'train'] for i in range(len(train_masks))]),
                      total=len(train_masks), desc='train'):
            pass


    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'test'] for i in range(len(val_masks))]),
                      total=len(val_masks),  desc='test'):
            pass

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'validation'] for i in range(len(val_masks))]),
                      total=len(val_masks),  desc='validation'):
            pass

    print(f'Preprocessing finished')