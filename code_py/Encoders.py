from tensorflow import keras


def make_Mobile(inputs):
    encoder = keras.applications.MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"]
    encoder_output = encoder.get_layer("block_16_project").output

    return skip_connection_names, encoder, encoder_output


def make_ResNet(inputs):
    encoder = keras.applications.ResNet50(input_tensor=inputs, weights="imagenet", include_top=False)
    skip_connection_names = ['input_image', 'conv1_relu', 'conv2_block3_out', 'conv3_block4_out']
    encoder_output = encoder.get_layer('conv4_block6_out').output

    return skip_connection_names, encoder, encoder_output
