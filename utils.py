# Numpy
import numpy as np

# Torch
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

num_classes = 8
h_crop = 256
w_crop = 256

# закодированные классы в бинарном виде и в one hot encoder
binary_encoded = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]]
one_hot_encoded = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1]]


# бинарный вид в one hot
def bin2ohe(mask, num_class, binary_encoded, one_hot_encoded):
    mask = mask.permute(1, 2, 0)
    mask = mask.numpy()
    mask = mask.astype(np.int64)
    h, w = mask.shape[:-1]
    layout = np.zeros((h, w, num_class), dtype=np.int64)
    for i, label in enumerate(binary_encoded):
        layout[np.all(mask == label, axis=-1)] = one_hot_encoded[i]
    layout = layout.astype(np.float64)
    layout = torch.from_numpy(layout)
    layout = layout.permute(2, 0, 1)
    return layout


# one hot вид в бинарный
def ohe2bin(mask, one_hot_encoded, binary_encoded):
    mask = mask.permute(1, 2, 0)
    mask = mask.numpy()
    h, w = mask.shape[:-1]
    layout = np.zeros((h, w, 3), dtype=np.int64)
    for i, label in enumerate(one_hot_encoded):
        layout[np.all(mask == label, axis=-1)] = binary_encoded[i]

    layout = layout.astype(np.float64)
    layout = torch.from_numpy(layout)
    layout = layout.permute(2, 0, 1)
    return layout


# Функция для подсчёта DICE коэффициента
def dice_coef(y_pred, y_true, classes):
    y_pred = tf.unstack(y_pred, axis=3)
    y_true = tf.unstack(y_true, axis=3)
    dice_summ = 0

    for i, (a_y_pred, b_y_true) in enumerate(zip(y_pred, y_true)):
        dice_calculate = (2 * tf.math.reduce_sum(a_y_pred * b_y_true) + 1) / \
                         (tf.math.reduce_sum(a_y_pred + b_y_true) + 1)

        dice_summ += dice_calculate
    avg_dice = dice_summ / classes
    return avg_dice


# Функция для подсчета DICE loss
def dice_loss(y_pred, y_true, classes):
    d_loss = 1 - dice_coef(y_pred, y_true, classes)
    return d_loss


# binary_crossentropy - дает хорошую сходимость модели при сбалансированном наборе данных
# DICE - хорошо в задачах сегментации но плохая сходимость
# Плохо сбалансированные данные, но хорошие реузьтаты:
# Binary crossentropy + 0.25 * DICE

def dice_bce_loss(y_pred, y_true, classes=1):
    total_loss = 0.25 * dice_loss(y_pred, y_true, classes) + tf.keras.losses.binary_crossentropy(y_pred, y_true)
    return total_loss


def img_show(*images, **kwargs):
    title = None
    if kwargs.get("titles", False):
        title = kwargs.pop("titles")

    fig, axs = plt.subplots(1, len(images), figsize=(18, 5))
    for idx, el in enumerate(zip(axs, images)):
        ax, img = el
        ax.imshow(img, **kwargs)
        if title:
            ax.set_title(title[idx])
