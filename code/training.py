import os
from os.path import exists

import numpy as np
from PIL import Image
from PIL.ImageOps import autocontrast
from keras.utils import load_img
from tensorflow import keras

import filter
import preprocessing
from UNet import make_U_model


# Configure the model for training.
# We use the "sparse" version of categorical-crossentropy because our target data is integers.
def train(model, train_gen, val_gen, epochs=15):
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint("segmentation.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)


def display_mask(mask, original, save_dir):
    """Quick utility to display a model's prediction.
    :param original: input image
    :param mask: prediction
    :param save_dir: comparison file path
    """
    mask = np.argmax(mask, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = autocontrast(keras.preprocessing.image.array_to_img(mask)).convert('RGBA')
    og = load_img(original).convert('RGBA')
    bg = Image.new('RGBA', (1600, 800), color='darkblue')
    bg.paste(og, (0, 0), img)
    bg.paste(img, (800, 0), img)
    bg.save(save_dir)


eval_bool = True


def init():
    input_dir = "./images/sorted/"
    target_dir = './images/quadmap/'
    out_dir = './images/result/'
    img_size = (800, 800)
    num_classes = 4
    batch_size = 4
    input_img_paths, target_img_paths = filter.load_dataset_images(input_dir, target_dir)

    print("Number of samples:", len(input_img_paths))

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    model = make_U_model(img_size, num_classes)
    if exists("./segmentation.h5"):
        model.load_weights("./segmentation.h5")

    train_gen, val_gen, val_paths = preprocessing.makeGens(input_img_paths, target_img_paths, batch_size, img_size)
    if not eval_bool:
        model.summary()
        train(model, train_gen, val_gen)

    val_preds = model.predict(val_gen)

    os.makedirs(out_dir, exist_ok=True)
    # Display mask predicted by our model
    for i in range(100):
        display_mask(val_preds[i], val_paths[0][i], out_dir + str(i) + ".png")


if __name__ == '__main__':
    init()
