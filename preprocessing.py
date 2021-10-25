import os
import numpy as np
import tensorflow as tf
import cv2

# Functions for importing and preprocessing the images
size = [500,500]
def image_reader(path, size):
    '''
    Reads images in a file folder in local system from a given path
    '''
    images = []
    for name in os.listdir(path):
        img = os.path.join(path, name)
        image = cv2.imread(img)
        image = tf.image.resize(image, size)
        images.append(cv2.imread(image))
    return images