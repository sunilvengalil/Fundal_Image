from keras import models, layers
from tensorflow.keras import backend as K

def conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(x)
    if batchnorm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(conv)
    if batchnorm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    return conv


def unetmodel(input_shape, dropout=0.2, batchnorm=True):
    # network structure
    filters = [16, 32, 64, 128, 256]
    kernelsize = 3
    upsample_size = 2

    inputs = layers.Input(input_shape)

    # Downsampling layers

    conv_128 = conv_block(inputs, kernelsize, filters[0], dropout, batchnorm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, kernelsize, filters[1], dropout, batchnorm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, kernelsize, filters[2], dropout, batchnorm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, kernelsize, filters[3], dropout, batchnorm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, kernelsize, filters[4], dropout, batchnorm)

    # Upsampling layers

    up_16 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, kernelsize, filters[3], dropout, batchnorm)
    # UpRes 7

    up_32 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, kernelsize, filters[2], dropout, batchnorm)
    # UpRes 8

    up_64 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, kernelsize, filters[1], dropout, batchnorm)
    # UpRes 9

    up_128 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, kernelsize, filters[0], dropout, batchnorm)

    conv_final = layers.Conv2D(1, kernel_size=(1, 1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    outputs = layers.Activation('sigmoid')(conv_final)

    # Model
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary()

    return model


def IoU_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def IoU_loss(y_true, y_pred):
    return -IoU_coef(y_true, y_pred)
