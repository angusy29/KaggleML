import cv2
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt
import pandas as pd

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# 120 breeds of dog
# 120 output nodes
NUM_DOG_TYPES = 120
BATCH_SIZE = 64
IMG_HEIGHT = 64
IMG_WIDTH = 64
LR = 0.5
TRAIN_DIR = 'train/'
CHANNELS = 3

MODEL_NAME = 'dog-breed-classification'

# processes labels.csv and returns a one_hot vector
def process_labels():
    nameToInt = {}
    file = open('dogtypes.txt', 'r')
    i = 0
    for line in file:
        line = line.rstrip()
        nameToInt[line] = i
        i += 1

    # open csv and parse through it, creating labels
    data = pd.read_csv('labels.csv', usecols=[0, 1], header=None)
    # one_hot = np.zeros(shape=(data.shape[0], NUM_DOG_TYPES))
    dict = {}
    for index, row in data.iterrows():
        dict[row[0]] = row[1]
        
    return dict, nameToInt

def create_label(img, imgIdToBreed, nameToInt):
    img = img[:-4]
    index = nameToInt[imgIdToBreed[img]]

    one_hot = np.zeros(shape=(NUM_DOG_TYPES))
    one_hot[index] = 1
    return one_hot

def create_train_data(imgIdToBreed, nameToInt):
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = cv2.resize(img_data, (IMG_HEIGHT, IMG_WIDTH))
        training_data.append([np.array(img_data), create_label(img, imgIdToBreed, nameToInt)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

imgIdToBreed, nameToInt = process_labels()

if not os.path.isfile('train_data.npy'):
    train_data = create_train_data(imgIdToBreed, nameToInt)
else:
    train_data = np.load('train_data.npy')

train = train_data[:-500]   # take everything up until the last 500
test = train_data[-500:]    # take the last 500

x_train = np.array([i[0] for i in train]).reshape(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
y_train = [i[1] for i in train]

x_test = np.array([i[0] for i in test]).reshape(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
y_test = [i[1] for i in test]

convnet = input_data(shape=[None, IMG_HEIGHT, IMG_WIDTH, CHANNELS], name='input')
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, NUM_DOG_TYPES, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.fit({'input': x_train}, {'targets': y_train}, n_epoch=10,
         validation_set=({'input': x_test}, {'targets': y_test}), 
         snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save("model.tfl")
