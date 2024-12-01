# Modelled after https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, Dropout, Cropping2D, Softmax, Input
import os
import matplotlib.pyplot as plt

def unet(input_size=(128, 128, 1), num_classes=2, base_level_filters=16, layers=3):
    # Contracting Path
    input_image = Input(input_size)

    last_layer = input_image
    expanding_layers = []
    for i in range(layers - 1):
        filters = base_level_filters * (2 ** i)
        conv1 = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'expand_conv_{}_1'.format(i))(last_layer)
        conv2 = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'expand_conv_{}_2'.format(i))(conv1)
        expanding_layers.append(conv2)
        last_layer = MaxPooling2D(name = 'pool_{}'.format(i))(conv2)
        last_layer = Dropout(0.3, name = 'expand_dropout_{}'.format(i))(last_layer)

    # Bottom Layer
    filters = base_level_filters * (2 ** (layers - 1))
    conv1 = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'bottom_conv_1')(last_layer)
    last_layer = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'bottom_conv_2')(conv1)

    # Expanding Path
    for i in range(layers - 2, -1, -1):
        filters = base_level_filters * (2 ** i)
        upconv = UpSampling2D(name = 'upconv_{}_1'.format(i))(last_layer)
        upconv = Conv2D(filters = filters, kernel_size = (2, 2), activation = 'relu', padding = 'same', name = 'upconv_{}_2'.format(i))(upconv)
        last_layer = concatenate([upconv, expanding_layers[i]], axis = 3, name = 'concat_{}'.format(i))

        last_layer = Dropout(0.3, name = 'contract_dropout_{}'.format(i))(last_layer)

        conv1 = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'contract_conv_{}_1'.format(i))(last_layer)
        conv2 = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'contract_conv_{}_2'.format(i))(conv1)
        last_layer = conv2

    conv_last = Conv2D(filters = num_classes, kernel_size = (1, 1), activation = 'softmax', name = 'conv_last')(last_layer)

    model = Model(inputs = input_image, outputs = conv_last, name = 'model')

    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy",
                  metrics="accuracy")
    return model

def get_data(data_folder, image_size = 128):
    image_folder = os.path.join(data_folder, 'reconstructions')
    mask_folder = os.path.join(data_folder, 'labels')

    image_files = os.listdir(image_folder)
    input_images = []
    input_masks = []

    print(len(image_files))

    for i in range(len(image_files)):
        image_path = os.path.join(image_folder, f'reconstruction_{i}.npy')
        mask_path = os.path.join(mask_folder, f'labels_{i}.npy')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        image = np.load(image_path)
        mask = np.load(mask_path)

        # Reshape to 128x128x1
        image = np.reshape(image, (image_size, image_size, 1))

        # Normalize image to 0, 1
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Reshape to 128x128x1
        # 1 -> background
        # 2 -> foreground
        mask = np.reshape(mask, (image_size, image_size, 1)) - 1
        
        input_images.append(image)
        input_masks.append(mask)

    input_array = np.array(input_images)
    gt_array = np.array(input_masks)
    return input_array, gt_array

def data_split(data):
    x, y = data
    x_test = x[0:100]
    x_train = x[100:]
    y_test = y[0:100]
    y_train = y[100:]
    return x_test, x_train, y_test, y_train   

def dsc(y_pred, y_test):
    y_test = np.squeeze(y_test)
    intersect = np.sum(y_pred * y_test, axis = (1, 2))
    union = np.sum(y_pred, axis = (1, 2)) + np.sum(y_test, axis = (1, 2))
    dsc = 2*intersect/union
    dsc[np.isnan(dsc)] = 1
    return dsc


def visualize_random(grid_size, reconstructions, masks, predictions, num_to_show=10):
    """
    Randomly select and visualize original and reconstructed images.
    """
    indices = np.random.choice(len(reconstructions), num_to_show, replace=False)
    plt.figure(figsize=(20, 15))

    for i, idx in enumerate(indices):
        reconstructed = reconstructions[idx].reshape(grid_size, grid_size)
        mask = masks[idx].reshape(grid_size, grid_size)
        pred = predictions[idx].reshape(grid_size, grid_size)

        # Reconstructed image
        plt.subplot(3, num_to_show, i + 1)
        plt.imshow(reconstructed, extent=(-1, 1, -1, 1), cmap="hot")
        plt.title(f"Reconstructed {idx}")
        plt.axis("off")

        # Mask
        plt.subplot(3, num_to_show, i + 1 + num_to_show)
        plt.imshow(mask, extent=(-1, 1, -1, 1), cmap="viridis")
        plt.title(f"Ground truth mask {idx}")
        plt.axis("off")

        # Prediction
        plt.subplot(3, num_to_show, i + 1 + num_to_show * 2)
        plt.imshow(pred, extent=(-1, 1, -1, 1), cmap="viridis")
        plt.title(f"Predicted mask {idx}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    