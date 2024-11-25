import os
from glob import glob
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

with open('../config.yaml', 'r') as file:
    paths_data = yaml.safe_load(file)['paths']

images_path = paths_data['data_path']
train_images_path = paths_data['train_images_path']
train_masks_path = paths_data['train_masks_path']
val_images_path = paths_data['val_images_path']
val_masks_path = paths_data['val_masks_path']
test_images_path = paths_data['test_images_path']
test_masks_path = paths_data['test_masks_path']

train_images = sorted(glob(os.path.join(images_path, train_images_path + '/**/*.jpg'), recursive=True),
                      key=lambda x: os.path.basename(x))
train_masks = sorted(glob(os.path.join(images_path, train_masks_path + '/**/*.bmp'), recursive=True),
                     key=lambda x: os.path.basename(x))
val_images = sorted(glob(os.path.join(images_path, val_images_path) + '/**/*.jpg', recursive=True),
                    key=lambda x: os.path.basename(x))
val_masks = sorted(glob(os.path.join(images_path, val_masks_path + '/**/*.bmp'), recursive=True),
                   key=lambda x: os.path.basename(x))
test_images = sorted(glob(os.path.join(images_path, test_images_path + '/**/*.jpg'), recursive=True),
                     key=lambda x: os.path.basename(x))
test_masks = sorted(glob(os.path.join(images_path, test_masks_path + '/**/*.bmp'), recursive=True),
                    key=lambda x: os.path.basename(x))


# welcome to the spaghetti ;)
def image_preprocessing(image, mask, copies=10, mode='train', ):
    image = datapoints.Image(read_image(image))
    mask = datapoints.Mask(transforms.RandomInvert(1)(transforms.ToImage()(Image.open(mask))))

    if mode == 'train':
        transformed_sizes = []

        for size in [128]:
            resize = transforms.Resize((size, size))
            input_and_mask_transform = transforms.Compose([transforms.RandomPerspective(.1),
                                                  transforms.RandomRotation(15)
                                                  ])

            input_transforms = transforms.Compose(
                [transforms.ColorJitter(brightness=0.2, hue=.05, contrast=0.2, saturation=0.2),
                 transforms.ElasticTransform(alpha=34, sigma=4),
                                                 transforms.ToDtype(torch.uint8)])

            crop_transforms = transforms.Compose([transforms.ColorJitter(brightness=0.1, hue=.05, contrast=0.1),
                                                  transforms.ElasticTransform(alpha=size / 2,
                                                                              sigma=2),
                                                  transforms.ToDtype(torch.uint8)])

            masks = []
            images = []
            images_cropped = []
            masks_cropped = []
            sizes = []

            images.append(resize(image))
            masks.append(resize(mask))
            image_cropped, mask_cropped = custom_crop(image, mask)
            image_cropped = resize(image_cropped)
            mask_cropped = resize(mask_cropped)
            images_cropped.append(image_cropped)
            masks_cropped.append(mask_cropped)

            sizes.append(size)

            true_mask = resize(image) != 0  # so zeroes stay zero

            # copies
            for i in range(copies - 1):
                t_image, t_mask = input_and_mask_transform(image, mask)
                image_cropped, mask_cropped = custom_crop(t_image, t_mask)
                image_cropped = resize(image_cropped)
                mask_cropped = resize(mask_cropped)
                t_img = input_transforms(resize(t_image)) * true_mask
                t_img_cropped = crop_transforms(image_cropped)

                images.append(t_img)
                masks.append(resize(t_mask))

                images_cropped.append(t_img_cropped)
                masks_cropped.append(mask_cropped)

                sizes.append(size)

            transformed_sizes.append([images, masks, images_cropped, masks_cropped, sizes])
        return transformed_sizes

    else:

        return (datapoints.Image(torch.unsqueeze(image, 0)), datapoints.Mask(torch.unsqueeze(mask, 0)),
                datapoints.Image(torch.unsqueeze(image_cropped, 0)), datapoints.Mask(torch.unsqueeze(mask_cropped, 0)))


def custom_crop(image, mask):
    rotation_transform = transforms.RandomRotation(20)
    degree_of_crop_random = 140
    image_rotated, mask_rotated = rotation_transform(image, mask)
    crop_coordinates = torchvision.ops.masks_to_boxes(mask_rotated).to(torch.int)[0]
    crop_coordinates_width_height = torch.tensor((crop_coordinates[1], crop_coordinates[0],
                                                  crop_coordinates[3] - crop_coordinates[1],
                                                  crop_coordinates[2] - crop_coordinates[0]))
    randomized = torch.randint(-20, degree_of_crop_random, (4,))
    crop_coordinates_width_height[2:] += randomized[2:]
    crop_coordinates_width_height[:2] -= randomized[:2]
    crop_coordinates_width_height[2:] += randomized[:2]
    mask_cropped = transforms.functional.crop(mask_rotated, *crop_coordinates_width_height)
    image_cropped = transforms.functional.crop(image_rotated, *crop_coordinates_width_height)
    return image_cropped, mask_cropped


#image_preprocessing(train_images[0], train_masks[0], 10, 128, mode='lol' )

def process_image(i):
    path = r'C:\my files\REFUGE\preprocessed'
    mode = i[1]
    i = i[0]

    if mode == 'train':
        original_image, original_label = train_images[i], train_masks[i]
        #if os.path.exists(fr'C:\my files\REFUGE\preprocessed\train\input\128\input_0_0.pt'):
        #    return

    elif mode == 'validation':
        original_image, original_label = val_images[i], val_masks[i]

        #if os.path.exists(os.path.join(path, fr'validation\masks\mask_{i}_0.bmp')):
        #    return

    elif mode == 'test':
        original_image, original_label = test_images[i], test_masks[i]
        #if os.path.exists(os.path.join(path, fr'\test\masks\mask_{i}_0.bmp')):
        #    return
    elif mode == 'train_only_resize':
        original_image, original_label = train_images[i], train_masks[i]

    sizes = [128]
    for size in sizes:
        if not os.path.exists(os.path.join(path, fr'{mode}\input\{size}')) or not os.path.exists(
                os.path.join(path, fr'{mode}\input_cropped\{size}')):
            os.makedirs(os.path.join(path, fr'{mode}\input\{size}'), exist_ok=True)
            os.makedirs(os.path.join(path, fr'{mode}\labels\{size}'), exist_ok=True)
            os.makedirs(os.path.join(path, fr'{mode}\input_cropped\{size}'), exist_ok=True)
            os.makedirs(os.path.join(path, fr'{mode}\labels_cropped\{size}'), exist_ok=True)

    if mode == 'train':
        processed_images = image_preprocessing(original_image, original_label, copies=10, mode=mode)

        for batch in processed_images:
            for j, (image, mask, image_cropped, mask_cropped, size) in enumerate(zip(*batch,)):

                values = torch.unique(mask)
                for idx, value in enumerate(values):  # note that the masks can actually be the same object from index 1
                    mask[mask == value] = idx
                    values = torch.unique(mask_cropped)
                    for idx, value in enumerate(values):
                        mask_cropped[mask_cropped == value] = idx

                    # Save tensors
                    torch.save(image, os.path.join(path, f'{mode}/input/{size}/input_{i}_{j}.pt'))
                    torch.save(mask, os.path.join(path, f'{mode}/labels/{size}/mask_{i}_{j}.pt'))
                    torch.save(image_cropped, os.path.join(path, f'{mode}/input_cropped/{size}/input_{i}_{j}.pt'))
                    torch.save(mask_cropped, os.path.join(path, f'{mode}/labels_cropped/{size}/mask_{i}_{j}.pt'))

                '''def tensor_to_rgb_image(tensor):
                    tensor = tensor.permute(1, 2, 0)  # Convert from [C, H, W] to [H, W, C] format for PIL
                    tensor = tensor.cpu().numpy().astype('uint8')  # Convert to numpy array and ensure dtype is uint8
                    return Image.fromarray(tensor)

                # Convert grayscale tensor to PIL image (assuming the tensor is in [H, W] format)
                def tensor_to_grayscale_image(tensor):
                    tensor = torch.squeeze(tensor)
                    tensor = tensor.cpu().numpy().astype('uint8')  # Convert to numpy array and ensure dtype is uint8
                    return Image.fromarray(tensor, mode='L')  # 'L' mode for grayscale image in PIL


                image_pil = tensor_to_rgb_image(image)
                mask_pil = tensor_to_grayscale_image(mask)  # Grayscale mask
                image_cropped_pil = tensor_to_rgb_image(image_cropped)
                mask_cropped_pil = tensor_to_grayscale_image(mask_cropped)  # Grayscale mask


                image_pil.save(os.path.join(path, f'{mode}/input/{size}/input_{i}_{j}.jpg'))
                mask_pil.save(os.path.join(path, f'{mode}/labels/{size}/mask_{i}_{j}.jpg'))
                image_cropped_pil.save(os.path.join(path, f'{mode}/input_cropped/{size}/input_{i}_{j}.jpg'))
                mask_cropped_pil.save(os.path.join(path, f'{mode}/labels_cropped/{size}/mask_{i}_{j}.jpg'))'''


    else:

        image, mask, image_cropped, mask_cropped = image_preprocessing(original_image, original_label, 5, mode=mode)
        for size in sizes:
            resize = transforms.Resize((size, size), antialias=True,
                                       interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

            image = resize(image).to(torch.float32)
            mask = resize(mask).to(torch.long)
            values = torch.unique(mask)
            for idx, value in enumerate(values):
                mask[mask == value] = idx

            torch.save(image, os.path.join(path, f'{mode}/input/{size}/input_{i}.pt'))
            torch.save(mask, os.path.join(path, f'{mode}/labels/{size}/mask_{i}.pt'))

            image = resize(image_cropped).to(torch.float32)
            mask = resize(mask_cropped).to(torch.long)
            values = torch.unique(mask)
            for idx, value in enumerate(values):
                mask[mask == value] = idx

            torch.save(image, os.path.join(path, f'{mode}/input_cropped/{size}/input_{i}.pt'))
            torch.save(mask, os.path.join(path, f'{mode}/labels_cropped/{size}/mask_{i}.pt'))


process_image((0, 'train'))

if __name__ == '__main__':
    freeze_support()

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'train'] for i in range(len(train_masks))]),
                      total=len(train_masks), desc='train'):
            pass
    """
    with Pool(processes=os.cpu_count()-1) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'validation'] for i in range(len(val_masks))]),
                      total=len(val_masks), desc='validation'):
            pass


    with Pool(processes=os.cpu_count()-1) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'test'] for i in range(len(val_masks))]),
                      total=len(val_masks), desc='test'):
            pass

   
    with Pool(processes=os.cpu_count()-1) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, [[i, 'train_only_resize'] for i in range(len(val_masks))]),
                      total=len(val_masks), desc='train only'):
            pass
    """
    print(f'Preprocessing finished')
