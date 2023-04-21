import os

import albumentations as abm
import numpy as np
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
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode='rgb')
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
    train_input_img_paths = input_img_paths[0]
    train_target_img_paths = target_img_paths[0]
    val_input_img_paths = input_img_paths[1]
    val_target_img_paths = target_img_paths[1]

    val_paths = (val_input_img_paths, val_target_img_paths)

    # Instantiate data Sequences for each split
    train_gen = DataGen(
        batch_size, img_size, train_input_img_paths, train_target_img_paths, aug
    )
    val_gen = DataGen(batch_size, img_size, val_input_img_paths, val_target_img_paths, aug)

    return train_gen, val_gen, val_paths


def pregenDataAugm(input_img_filenames, target_img_filenames, in_path, tg_path, dest_path):
    counter = len(input_img_filenames)-1

    out_path = dest_path
    os.makedirs(out_path + 'train/input/', exist_ok=True)
    os.makedirs(out_path + 'train/maps/', exist_ok=True)
    os.makedirs(out_path + 'val/input/', exist_ok=True)
    os.makedirs(out_path + 'val/maps/', exist_ok=True)

    for input, target in zip(input_img_filenames, target_img_filenames):

        out_path = dest_path + 'train'
        image = load_img(os.path.join(in_path+"/white/", input), target_size=(800, 800), color_mode='rgb')
        mask = load_img(os.path.join(tg_path, target), target_size=(800, 800), color_mode="grayscale")

        # save originals
        og_image = image
        og_mask = mask
        image.save(out_path + "/input/" + input)
        mask.save(out_path + "/maps/" + target)

        # step 1 - horiz flip
        counter += 1
        transform = abm.Compose([abm.HorizontalFlip(p=1)])(image=asarray(og_image), mask=asarray(og_mask))
        hoz_flip = transform['image']
        mask = transform['mask']
        Image.fromarray(hoz_flip).save(out_path + "/input/" + str(counter) + '.png')
        Image.fromarray(mask).save(out_path + "/maps/" + str(counter) + '.png')

        # step 2 - vert flip
        counter += 1
        transform = abm.Compose([abm.VerticalFlip(p=1)])(image=asarray(og_image), mask=asarray(og_mask))
        ver_flip = transform['image']
        mask = transform['mask']
        Image.fromarray(ver_flip).save(out_path + "/input/" + str(counter) + '.png')
        Image.fromarray(mask).save(out_path + "/maps/" + str(counter) + '.png')

        # step 3 - rotate 180
        counter += 1
        transform = abm.Compose([abm.Affine(rotate=[180, 180], p=1)])(image=asarray(og_image), mask=asarray(og_mask))
        rot_180_image = transform['image']
        mask = transform['mask']
        Image.fromarray(rot_180_image).save(out_path + "/input/" + str(counter) + '.png')
        Image.fromarray(mask).save(out_path + "/maps/" + str(counter) + '.png')

        # step 4 - rotate -90
        counter += 1
        transform = abm.Compose([abm.Affine(rotate=[-90, -90], p=1)])(image=asarray(og_image), mask=asarray(og_mask))
        rot_90_image = transform['image']
        mask = transform['mask']
        Image.fromarray(rot_90_image).save(out_path + "/input/" + str(counter) + '.png')
        Image.fromarray(mask).save(out_path + "/maps/" + str(counter) + '.png')

        # step 5 - rotate -90 + hoz flip
        counter += 1
        transform = abm.Compose([abm.HorizontalFlip(p=1)])(image=asarray(rot_90_image), mask=asarray(mask))
        rot_90_image_flip_1 = transform['image']
        mask = transform['mask']
        Image.fromarray(rot_90_image_flip_1).save(out_path + "/input/" + str(counter) + '.png')
        Image.fromarray(mask).save(out_path + "/maps/" + str(counter) + '.png')

        # step 6 - rotate 90 + hoz flip
        counter += 1
        transform = abm.Compose([abm.Affine(rotate=[90, 90], p=1), abm.HorizontalFlip(p=1)])(image=asarray(og_image), mask=asarray(og_mask))
        rot_90_image_flip_2 = transform['image']
        mask = transform['mask']
        Image.fromarray(rot_90_image_flip_2).save(out_path + "/input/" + str(counter) + '.png')
        Image.fromarray(mask).save(out_path + "/maps/" + str(counter) + '.png')

        # step val - rotate 90 on original for valset
        out_path = dest_path+'val/'
        transform = abm.Compose([abm.Affine(rotate=[90, 90], p=1)])(image=asarray(og_image), mask=asarray(og_mask))
        image = transform['image']
        mask = transform['mask']
        Image.fromarray(image).save(out_path + "/input/" + input)
        Image.fromarray(mask).save(out_path + "/maps/" + target)


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
