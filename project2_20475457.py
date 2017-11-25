#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:40:47 2017

@author: llr
"""


import os
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing the directory of an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    f = open(data_dir)
    paths = f.readlines()
    f.close()

    labels = []
    images = []
    for line in paths:
        im, lab = line.split(' ')
        images.append(im)
        labels.append(int(lab.rstrip()))

    return images, labels


# Load training and testing datasets.
TRAIN_PATH = '/train.txt'
VAL_PATH = '/val.txt'
TEST_PATH = '/test.txt'
work_dir = os.getcwd()
data_dir = work_dir
train_data_dir = data_dir + TRAIN_PATH
val_data_dir = data_dir + VAL_PATH
test_data_dir = data_dir + TEST_PATH

images, labels = load_data(train_data_dir)

# Read the photos
Images = []
for i in images:
    Images.append(skimage.data.imread(i))


# Resize images
images32 = [skimage.transform.resize(image, (32, 32)) for image in Images]


    
labels_a = np.array(labels)
images_a = np.array(images32)

# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    
#2 conv layer，but did not work
# =============================================================================
#     inputs = tf.reshape(images_ph, (-1, 32, 32, 3))
# 
#     #conv1
#     conv1 = tf.contrib.layers.conv2d(inputs, 4, [5, 5], scope='conv_layer1', activation_fn=tf.nn.tanh);
#     pool1 = tf.contrib.layers.max_pool2d(conv1, [2, 2], padding='SAME')
#     #conv2
#     conv2 = tf.contrib.layers.conv2d(pool1, 6, [5, 5], scope='conv_layer2', activation_fn=tf.nn.tanh);
#     pool2 = tf.contrib.layers.max_pool2d(conv2, [2, 2], padding='SAME')
#     pool2_shape = pool2.get_shape()
#     pool2_in_flat = tf.reshape(pool2, [pool2_shape[0].value or -1, np.prod(pool2_shape[1:]).value])
#     #fc
#     logits = tf.contrib.layers.fully_connected(pool2_in_flat, 1024, scope='fc_layer1', activation_fn=tf.nn.relu)
# =============================================================================


    images_flat = tf.contrib.layers.flatten(images_ph)

    #Fully connected layer. 
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(images_flat, 30, tf.nn.relu)

    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()

#print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)


# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
_ = session.run([init])

# Training process
for i in range(200):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 20 == 0:
        print("Loss: ", loss_value)

# Validation

f = open(test_data_dir)
paths2 = f.readlines()
f.close()

labels = []
images2 = []
for line in paths2:
    im = line
    images2.append(im.rstrip())

image_val2 = []
for v2 in images2:
    image_val2.append(skimage.data.imread(v2))

# Resize val images
image_val32 = [skimage.transform.resize(image, (32, 32)) for image in image_val2]
#display_images_and_labels(image_val64, label_val)

# Run the "predicted_labels" op.
predicted = session.run([predicted_labels], feed_dict={images_ph: image_val32})[0]
#print(predicted)

# Calculate how many matches we got.
#match_count = sum([int(y == y_) for y, y_ in zip(label_val, predicted)])
#accuracy = match_count / len(label_val)
#print("Accuracy: {:.3f}".format(accuracy))

np.savetxt('project2_20475457.txt',np.array(predicted)) 
