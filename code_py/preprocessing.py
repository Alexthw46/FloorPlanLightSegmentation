import os
import random

import numpy as np
import albumentations as abm
from PIL import Image
from keras.utils import load_img, Sequence
from numpy import asarray


class DataGen(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, aug):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.aug = aug

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        augmentation = abm.Compose([
            abm.RandomRotate90(p=0.3),
            abm.HorizontalFlip(p=0.3),
            abm.VerticalFlip(p=0.3)
        ])
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (4,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode='rgba')
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3, 4. Subtract to get 0, 1, 2, 3:
            y[j] -= 1
            if self.aug:
                transformed = augmentation(image=x[j], mask=y[j])
                x[j], y[j] = transformed['image'], transformed['mask']
        return x, y


def makeGens(input_img_paths, target_img_paths, batch_size, img_size, val_split=0.2, aug=False):
    # Split our img paths into a training and a validation set
    val_samples = int(len(input_img_paths) * val_split)
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    val_paths = (val_input_img_paths, val_target_img_paths)

    # Instantiate data Sequences for each split
    train_gen = DataGen(
        batch_size, img_size, train_input_img_paths, train_target_img_paths, aug
    )
    val_gen = DataGen(batch_size, img_size, val_input_img_paths, val_target_img_paths, aug)

    return train_gen, val_gen, val_paths


def pregenDataAugm(input_img_filenames, target_img_filenames, in_path, tg_path):
    augmentation = abm.Compose([
        abm.RandomRotate90(p=0.3),
        abm.HorizontalFlip(p=0.3),
        abm.VerticalFlip(p=0.3)
    ])
    counter = len(input_img_filenames)
    for input, target in zip(input_img_filenames, target_img_filenames):
        image = load_img(os.path.join(in_path, input), target_size=(800, 800), color_mode='rgba')
        mask = load_img(os.path.join(tg_path, target), target_size=(800, 800), color_mode="grayscale")

        # save originals
        image.save(in_path + "/aug/" + input)
        mask.save(tg_path + "/aug/" + target)

        # step 1 - horiz flip
        counter += 1
        transform = abm.Compose([abm.HorizontalFlip(p=1)])(image=asarray(image), mask=asarray(mask))
        image = transform['image']
        mask = transform['mask']
        Image.fromarray(image).save(in_path + "/aug/" + str(counter) + '.png')
        Image.fromarray(mask).save(tg_path + "/aug/" + str(counter) + '.png')

        # step 2 - vert flip
        counter += 1
        transform = abm.Compose([abm.VerticalFlip(p=1)])(image=asarray(image), mask=asarray(mask))
        image = transform['image']
        mask = transform['mask']
        Image.fromarray(image).save(in_path + "/aug/" + str(counter) + '.png')
        Image.fromarray(mask).save(tg_path + "/aug/" + str(counter) + '.png')

        # step 3 - rotate 90
        counter += 1
        transform = abm.Compose([abm.RandomRotate90(p=1)])(image=asarray(image), mask=asarray(mask))
        image = transform['image']
        mask = transform['mask']
        Image.fromarray(image).save(in_path + "/aug/" + str(counter) + '.png')
        Image.fromarray(mask).save(tg_path + "/aug/" + str(counter) + '.png')


# convert the rgba image to a grayscale map [(0, background), (1, walls), (2, rooms), (3, lights)]
# noinspection PyUnresolvedReferences
def imageToMap(in_path, out_path, filename):
    image = Image.open(os.path.join(in_path, filename))
    bg = Image.new('RGBA', (800, 800), color=(0, 0, 0, 0))
    bg.paste(image, None, image)

    newImage = bg.convert('LA')
    pixdata = newImage.load()

    width, height = newImage.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y][1] == 0:
                pixdata[x, y] = (1, 255)
            elif pixdata[x, y][0] == 0:
                pixdata[x, y] = (2, 255)
            elif pixdata[x, y][0] == 255:
                pixdata[x, y] = (3, 255)
            else:
                pixdata[x, y] = (4, 255)

    newImage.save(out_path + filename)
