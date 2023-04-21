import os

import cv2


def load_dataset_images(input_dir):

    train_img_paths = sorted(
        [
            os.path.join(input_dir+'train/input', fname)
            for fname in os.listdir(input_dir+'train/input')
            if fname.endswith(".png")
        ]
    )
    train_maps_paths = sorted(
        [
            os.path.join(input_dir+'train/maps', fname)
            for fname in os.listdir(input_dir+'train/maps')
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    val_img_paths = sorted(
        [
            os.path.join(input_dir+'val/input', fname)
            for fname in os.listdir(input_dir+'val/input')
            if fname.endswith(".png")
        ]
    )
    val_maps_paths = sorted(
        [
            os.path.join(input_dir+'val/maps', fname)
            for fname in os.listdir(input_dir+'val/maps')
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    return (train_img_paths, val_img_paths), (train_maps_paths, val_maps_paths)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(filename)
    images.sort()
    return images

