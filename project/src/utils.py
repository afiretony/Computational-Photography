import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def load_img(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def display_img(im):
    im_copy = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", im_copy)
    cv2.waitKey(0)


def visualize_depth(im, depth):
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(121)
    ax1.margins(0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_axis_off()
    ax1 = plt.imshow(im)

    ax2 = plt.subplot(122)
    ax2.margins(0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_axis_off()
    ax2 = plt.imshow(depth, cmap="gray")

    plt.show()


def save_depth(im, depth, filepath):
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(121)
    ax1.margins(0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_axis_off()
    ax1 = plt.imshow(im)

    ax2 = plt.subplot(122)
    ax2.margins(0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_axis_off()
    ax2 = plt.imshow(depth, cmap="gray")
    plt.savefig(filepath)
