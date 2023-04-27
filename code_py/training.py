import argparse
import os
from os.path import exists

from code_py import DeepLab

os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
from PIL import Image
from PIL.ImageOps import autocontrast
from keras.utils import load_img
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm

import filter
import preprocessing
import UNet

import keras.backend as K

from keras.losses import Loss


class WeightedDiceLoss(Loss):
    def __init__(self, weights, smooth=1, name='weighted_dice_loss'):
        super().__init__(name=name)
        self.weights = weights
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=tf.shape(y_pred)[-1])
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        intersection = K.sum(y_true * y_pred * self.weights, axis=[1, 2])
        union = K.sum((y_true + y_pred) * self.weights, axis=[1, 2])
        return 1 - K.mean((2. * intersection + self.smooth) / (union + self.smooth), axis=0)


# Configure the model for training.
# We use the "sparse" version of categorical-crossentropy because our target data is integers.
def train(model: keras.Model, args, train_gen: preprocessing.DataGen, val_gen: preprocessing.DataGen,
          epochs: int = 25) -> None:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.001),
                  loss=WeightedDiceLoss(weights=[0.5, 1.0, 2.0, 4.0]),
                  metrics=[
                      keras.metrics.IoU(name="iou", num_classes=4, target_class_ids=[0, 1, 2], sparse_y_pred=False),
                      keras.metrics.IoU(name="iou_lights", num_classes=4, target_class_ids=[3], sparse_y_pred=False)
                  ], jit_compile=False)

    callbacks = [
        keras.callbacks.TensorBoard(log_dir="tensorboard_dc_" + args.decoder + "_" + args.backbone),
        keras.callbacks.ModelCheckpoint("segmentation_dc_" + args.decoder + "_" + args.backbone + ".h5",
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='iou_lights',
            factor=0.1,
            patience=10,
            verbose=1,
            mode='max',
            min_delta=0.05,
            cooldown=0,
            min_lr=1e-10
        ), keras.callbacks.EarlyStopping(monitor="iou_lights", min_delta=0.001, patience=10, verbose=1,
                                         mode="max", restore_best_weights=True, start_from_epoch=5)
    ]

    # Train the model, doing validation at the end of each epoch.
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks, use_multiprocessing=True)


def display_mask(masks, original, save_dir):
    """Quick utility to display a model's prediction.
    :param original: input image
    :param masks: predictions
    :param save_dir: comparison file path
    """
    for i in range(len(masks)):
        mask = masks[i]
        mask = np.argmax(mask, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = autocontrast(keras.preprocessing.image.array_to_img(mask)).convert('RGBA').resize((800, 800))
        og = load_img(original[0][i], color_mode='rgba')
        og_quad = autocontrast(load_img(original[1][i], color_mode='grayscale')).convert('RGBA')
        bg = Image.new('RGBA', (2400, 800), color=(0, 0, 0, 0))
        bg.paste(og, (0, 0))
        bg.paste(og_quad, (800, 0))
        bg.paste(img, (1600, 0))
        bg.save(save_dir + original[0][i][19:])


def init(args):
    data_dir = args.input
    out_dir = './images/train_result_dc_' + args.decoder + '_' + args.backbone + '/'
    img_size = (512, 512)
    num_classes = 4
    batch_size = 2
    input_img_paths, target_img_paths = filter.load_dataset_images(data_dir)

    print("Number of samples:", len(input_img_paths[0]))

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    if args.backbone == "unet":
        model = UNet.make_U_model(img_size, num_classes)
        weight_name = "./segmentation.h5"
    else:
        if args.decoder == "linknet":
            model = sm.Linknet(backbone_name=args.backbone, input_shape=img_size + (3,), classes=4,
                               activation='softmax')
        elif args.decoder == "fpn":
            model = sm.FPN(backbone_name=args.backbone, input_shape=img_size + (3,), classes=4, activation='softmax',
                           pyramid_dropout=0.7)
        elif args.decoder == "psp":
            model = sm.PSPNet(backbone_name=args.backbone, input_shape=img_size + (3,), classes=4, activation='softmax',
                              psp_dropout=0.5)
        elif args.decoder == "deeplab":
            model = DeepLab.DeeplabV3Plus(image_size=img_size[0], num_classes=4)
        else:
            model = sm.Unet(backbone_name=args.backbone, input_shape=img_size + (3,), classes=4, activation='softmax')
            '''base_model_input = model.input base_model_output = model.get_layer('decoder_stage4b_relu').output #add 
            dropout base_model_output = keras.layers.Dropout(0.5)(base_model_output) output = keras.layers.Conv2D(
            filters= num_classes, kernel_size=(3, 3), padding='same', use_bias=True, 
            kernel_initializer='glorot_uniform', name='final_conv', activation = 'softmax')(base_model_output) model 
            = keras.models.Model(base_model_input, output)'''
        weight_name = "./segmentation_dc_" + args.decoder + "_" + args.backbone + ".h5"

    if exists(weight_name):
        model.load_weights(weight_name)

    train_gen, val_gen, val_paths = preprocessing.makeGens(input_img_paths, target_img_paths, batch_size, img_size,
                                                           aug=args.augmentation)
    if args.train:
        model.summary()
        train(model, args, train_gen, val_gen, epochs=args.epochs)

    val_preds = model.predict(val_gen)

    os.makedirs(out_dir, exist_ok=True)

    # Display mask predicted by our model
    display_mask(val_preds, val_paths, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=False,
        action='store_true',
        help="Flag for training",
    )
    parser.add_argument(
        "--input",
        type=str,
        metavar="input",
        default="./images/",
        help="Path of the input images",
    )
    parser.add_argument(
        "--target",
        type=str,
        metavar="target",
        default="./images/maps/",
        help="Path of the annotation images",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        metavar="epochs",
        default=15,
        help="Number of epochs",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        metavar="encoder",
        default="mobilenetv2",
        help="Set backbone encoder",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        default="unet",
        help="Set decoder"
    )

    parser.add_argument(
        "--augmentation",
        type=bool,
        metavar="dataugmentation",
        default=False,
        help="Toggle pipeline augmentation",
    )
    init(parser.parse_args())
