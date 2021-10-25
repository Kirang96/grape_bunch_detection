# Contains functions that is used to import data, create input pipeline, plot and preprocess the images


import os
import cv2
import tensorflow as tf
import pandas as pd
import sys
import random
import numpy as np
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt


from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


BATCH_SIZE = 32


def file_name_collector(path):
    '''
    First the names of files are collected using this function.
    '''
    full_file_names = []
    valid_images = (".jpg",".jpeg",".png")
    for filenames in os.listdir(path):
        if filenames.endswith(valid_images):
            full_file_names.append(filenames)
    return full_file_names


def dataset_import(required_filenames, path):
    '''
    This function can be used to import dataset images when a filepath to the folder containing images is given. 
    The output will be a list.
    '''
    images = []
    image_count = 0

    for filenames in required_filenames:
        image_path = os.path.join(path, filenames)
        img = cv2.imread(image_path)
        image_count = image_count + 1
        img = tf.image.resize(img, [640,640])
        images.append(img)
        sys.stdout.write('\rFile read: %d' %image_count)
        sys.stdout.flush()        
    return images

def test_dataset_import(required_filenames, path):
    '''
    This function can be used to import dataset images when a filepath to the folder containing images is given. 
    The output will be a list.
    '''
    images = []
    image_count = 0

    for filenames in required_filenames:
        image_path = os.path.join(path, filenames)
        img = cv2.imread(image_path)
        image_count = image_count + 1
        img = tf.image.resize(img, [640,640])
        images.append(np.expand_dims(img, axis=0))
        sys.stdout.write('\rFile read: %d' %image_count)
        sys.stdout.flush()        
    return images



def create_dataset(image, mask):
    '''
    Creates a tensorflow dataset object which can be used in models
    '''
    training_dataset = tf.data.Dataset.from_tensor_slices((image, mask))

    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.cache()
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return training_dataset


  # Parsing the XML file
def xml_parse_to_csv(path):
    '''
    This function takes a path and converts the necessary tags to a cvs files. It will return the csv files and saves the csv to local as well.
    This function is created using xml_to_csv.py from datitran's github: https://github.com/datitran/raccoon_dataset.
    '''
    rows=[]
    for label_names in os.listdir(path):
        if label_names.endswith(".xml"):
            label_file = os.path.join(path, label_names)
            xmlparse = ET.parse(label_file)
            root = xmlparse.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                        )
            rows.append(value)
    column_name = ['xmin', 'ymin', 'xmax', 'ymax', 'label']
    labels = pd.DataFrame(rows, columns=column_name)
    return labels


# Function to plot the bounding boxes on images

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.3)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)