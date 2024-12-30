import os
import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image

np.random.seed(0)
from sklearn.model_selection import train_test_split
# from arkinas_model import unetmodel
from tensorflow.keras.optimizers import Adam
# from arkinas_model import IoU_coef, IoU_loss

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


image_dataset = []
mask_dataset = []


def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized


path1 = '/Users/sunilv/gitprojects/medical_image_processing/fundal_image/datasets/drive/training/images'  # training images directory
path2 = '/Users/sunilv/gitprojects/medical_image_processing/fundal_image/datasets/drive/training/images'  # training masks directory

patch_size = 512

images = sorted(os.listdir(path1))
for i, image_name in enumerate(images):
    if image_name.endswith(".jpg"):

        image = skimage.io.imread(path1 + "/" + image_name)  # Read image
        image = image[:, :, 1]  # selecting green channel
        image = clahe_equalized(image)  # applying CLAHE
        SIZE_X = (image.shape[1] // patch_size) * patch_size  # getting size multiple of patch size
        SIZE_Y = (image.shape[0] // patch_size) * patch_size  # getting size multiple of patch size
        image = Image.fromarray(image)
        image = image.resize((SIZE_X, SIZE_Y))  # resize image
        image = np.array(image)

        patches_img = patchify(image, (patch_size, patch_size),
                               step=patch_size)  # create patches(patch_sizexpatch_sizex1)

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                single_patch_img = (single_patch_img.astype('float32')) / 255.
                image_dataset.append(single_patch_img)

print(SIZE_X, SIZE_Y, len(images))

masks = sorted(os.listdir(path2))
for i, mask_name in enumerate(masks):
    if mask_name.endswith(".jpg"):

        mask = skimage.io.imread(path2 + "/" + mask_name)  # Read masks
        SIZE_X = (mask.shape[1] // patch_size) * patch_size  # getting size multiple of patch size
        SIZE_Y = (mask.shape[0] // patch_size) * patch_size  # getting size multiple of patch size
        mask = Image.fromarray(mask)
        mask = mask.resize((SIZE_X, SIZE_Y))  # resize image
        mask = np.array(mask)

        patches_mask = patchify(mask, (patch_size, patch_size),
                                step=patch_size)  # create patches(patch_sizexpatch_sizex1)

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                single_patch_mask = (single_patch_mask.astype('float32')) / 255.
                mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

image_dataset = np.expand_dims(image_dataset, axis=-1)
mask_dataset = np.expand_dims(mask_dataset, axis=-1)

IMG_HEIGHT = patch_size
IMG_WIDTH = patch_size
IMG_CHANNELS = 1
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = unetmodel(input_shape)
model.compile(optimizer=Adam(lr=1e-3), loss=IoU_loss, metrics=['accuracy', IoU_coef])

# splitting data into 70-30 ratio to validate training performance
x_train, x_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.3, random_state=0)

# train model
history = model.fit(x_train, y_train,
                    verbose=1,
                    batch_size=16,
                    validation_data=(x_test, y_test),
                    shuffle=False,
                    epochs=150)

# training-validation loss curve
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure(figsize=(7, 5))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# training-validation accuracy curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.figure(figsize=(7, 5))
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'y', label='Validation Accuracy')
plt.title('Training and validation accuracies')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

# training-validation IoU curve
iou_coef = history.history['IoU_coef']
val_iou_coef = history.history['val_IoU_coef']
plt.figure(figsize=(7, 5))
plt.plot(epochs, iou_coef, 'r', label='Training IoU')
plt.plot(epochs, val_iou_coef, 'y', label='Validation IoU')
plt.title('Training and validation IoU coefficients')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

# save model
# model.save('/content/drive/MyDrive/training/retina_Unet_150epochs.hdf5')