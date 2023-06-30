import os
from glob import glob


def get_preprocessed_images_paths():
    path = r'C:\my files\REFUGE\3\train'

    train_masks = sorted(glob(os.path.join(path, r'masks/**/*.bmp'), recursive=True))
    train_images = sorted(glob(os.path.join(path, r'image/**/*.jpg'), recursive=True))

    path = r'C:\my files\REFUGE\3\validation'

    val_masks = sorted(glob(os.path.join(path, r'masks/**/*.bmp'), recursive=True))
    val_images = sorted(glob(os.path.join(path, r'image/**/*.jpg'), recursive=True))

    path = r'C:\my files\REFUGE\3\test'

    test_masks = sorted(glob(os.path.join(path, r'masks/**/*.bmp'), recursive=True))
    test_images = sorted(glob(os.path.join(path, r'image/**/*.jpg'), recursive=True))

    return train_images, train_masks, val_images, val_masks, test_images, test_masks
