# grape_bunch_detection

## Summary

Grape detection project was created by fine tuning a pretrained model using Tensorflow. It detects grape bunches in an image and draws bounding boxes around it.
I created this with Tensorflow object detection API and its example codes.

## Project structure

### Importing necessary packages

I have used Tensorflow object detection API in this project for visualization and building the model.
Tensorflow object detection API was first installed by cloning their github and installed using python.
These were also imported to the main file

### Data collection

This model was pretrained on COCO dataset and fine tuned using images that I have collected from internet. 
15 Images used were downloaded from Google and rescaled to 640x640 before inputing to the model.
These were loaded using the function I created and converted to tensor format.

### Annotating images

LabelImg tool was used to annotate the grape bunch images in PASCAL VOC format. The annotations were saved in XML file format and then read in python and converted to CSV. These were used as numpy arrays in the model.
I have created a function that takes a path and converts the necessary tags to a cvs files. It will return the csv files and saves the csv to local as well.
This function is created using xml_to_csv.py from datitran's github: https://github.com/datitran/raccoon_dataset.

### Preprocessing

During preprocessing the images and ground truth boxes were converted to tensor format. Classes were one hot encoded and then converted to tensors. This is the standard input for pretrained model on TFOD. Images and annotations could also be input in TFRECORD format.

### Model building

This project uses a pretrained RetinaNet model trained on COCO dataset. The model was taken from TFOD Model garden. It was loaded from checkpoint by configuring the model config file. Weights were taken from Tensorflow model zoo. These weights were loaded to the model and then trained on the data that I had collected from that checkpoint.
The model uses an Adam optimizer with 0.01 learning rate. 
A custom training loop was built using tensorflow Gradient tape. It uses model loss function which calculates the classification and localization loss.
Model was then trained for 5000 steps on GPU. Nvidea CUDA was leveraged to make the training faster.
Model was then tested on 5 test images as well.

## Technologies used

- Python
- Tensorflow
- Tensorflow object detection API
- LabelImg
- Pandas
- Matplotlib
- VSCODE as IDE
- Numpy
