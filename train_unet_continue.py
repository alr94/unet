# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description='Run CNN training on patches with' 
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-m', '--model', help = 'Input model weights')
parser.add_argument('-g', '--gpu',    help = 'Which GPU index', default = '0')
parser.add_argument('-l', '--loss',    help = 'Loss goal', default = '0.7')

args = parser.parse_args()

################################################################################
# setup tensorflow enviroment variables
import os
from os.path import exists, isfile, join
os.environ['KERAS_BACKEND']        = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

################################################################################
# setup tensorflow and keras
import tensorflow as tf
print ('Using Tensorflow version: ', tf.__version__)

import keras
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from keras.preprocessing.image import *
print ('Using Keras version: ', keras.__version__)
keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')
        
################################################################################
# Other setups
import numpy as np
import itertools
import json
from utils import read_config, get_patch_size, count_events
import math
import datetime
from collections import defaultdict
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

################################################################################
# Additional utils
def save_model(model, name):
  try:
    name += '_'
    name += datetime.datetime.now().strftime("%y%m%d-%H:%M")
    with open(name + '_architecture.json', 'w') as f: f.write(model.to_json())
    model.save_weights(name + '_weights.h5', overwrite=True)
    return True
  
  except: 
    return False
  
################################################################################
# Model utilities
def intersection(true, pred):
  intersection = K.sum(K.abs(true * pred), axis= [1, 2])
  return intersection

def sum_(true, pred):
  sum_ = K.sum(K.abs(true) + K.abs(pred), axis= [1, 2])
  return sum_

def jaccard(true, pred):
  smooth = 1e-10
  i      = intersection(true, pred)
  s      = sum_(true, pred)
  u      = s - i
  jac    = (i + smooth) / (u + smooth)
  jac    = jac * tf.cast(jac <= 1., jac.dtype)
  return jac

def loss_jaccard(true, pred):
  jac = jaccard(true, pred)
  return - jac

################################################################################
# Get data
dataTypes = ['wire', 'cnn', 'truth']
data      = defaultdict()

for dataType in dataTypes: 
  data[dataType] = np.load(dataType + '/' + dataType + '.npy')

n_patches = data['truth'].shape[0]
patch_w   = data['truth'].shape[1]
patch_h   = data['truth'].shape[2]

x_data = data['wire']
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))

y_data = data['truth']
y_data = y_data.reshape((y_data.shape[0], y_data.shape[1], y_data.shape[2], 1))

print (x_data.shape)

################################################################################ 
# Testing for loss functions, adn filtering data
r = loss_jaccard(K.variable(y_data), 
                 K.variable(y_data)).eval(session = K.get_session())
 
idx = np.any(r < -1e-10, axis=1)
x_data_filtered = x_data[idx]
y_data_filtered = y_data[idx]
print (x_data_filtered.shape)
  
################################################################################
# Data augmentation
num_samples     = x_data_filtered.shape[0]
batch_size      = 16
rounds          = 1
steps_per_epoch = math.ceil(rounds * num_samples/batch_size)
seed            = 1

# data_gen_args = dict(featurewise_center            = True,
#                      featurewise_std_normalization = True,
#                      rotation_range                = 90,
#                      width_shift_range             = 0.1,
#                      height_shift_range            = 0.1)
# 
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen  = ImageDataGenerator(**data_gen_args)
# 
# image_datagen.fit(x_data_filtered, augment = True, rounds = rounds, seed = seed)
# mask_datagen.fit(y_data_filtered, augment = True, rounds =rounds, seed = seed)
# 
# image_generator = image_datagen.flow(x_data_filtered, shuffle = False, 
#                                      seed = seed, batch_size = batch_size)
# mask_generator  = mask_datagen.flow(y_data_filtered, shuffle = False, 
#                                     seed = seed, batch_size = batch_size)
# 
# train_generator = itertools.izip(image_generator, mask_generator)

################################################################################
# Model definition 
sess = tf.InteractiveSession()
with sess.as_default():
  
  ##############################################################################
  # Input
  main_input = Input(shape=(patch_w, patch_h, 1), name='main_input')
  
  ##############################################################################
  # Downscaling
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(main_input)
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
  
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
  
  ##############################################################################
  # Upscaling
  up6    = UpSampling2D(size = (2, 2))(conv5)
  up6    = Conv2D(512, 2, activation = 'relu', padding = 'same')(up6)
  merge6 = concatenate([conv4, up6], axis = 3)
  drop6  = Dropout(0.5)(merge6)
  conv6  = Conv2D(512, 3, activation = 'relu', padding = 'same')(drop6)
  conv6  = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
  
  up7    = UpSampling2D(size = (2, 2))(conv6)
  up7    = Conv2D(256, 2, activation = 'relu', padding = 'same')(up7)
  merge7 = concatenate([conv3, up7], axis = 3)
  drop7  = Dropout(0.5)(merge7)
  conv7  = Conv2D(256, 3, activation = 'relu', padding = 'same')(drop7)
  conv7  = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
  
  up8    = UpSampling2D(size = (2, 2))(conv7)
  up8    = Conv2D(128, 2, activation = 'relu', padding = 'same')(up8)
  merge8 = concatenate([conv2, up8], axis = 3)
  drop8  = Dropout(0.5)(merge8)
  conv8  = Conv2D(128, 3, activation = 'relu', padding = 'same')(drop8)
  conv8  = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
  
  up9    = UpSampling2D(size = (2, 2))(conv8)
  up9    = Conv2D(64, 2, activation = 'relu', padding = 'same')(up9)
  merge9 = concatenate([conv1, up9], axis = 3)
  drop9  = Dropout(0.5)(merge9)
  conv9  = Conv2D(64, 3, activation = 'relu', padding = 'same')(drop9)
  conv9  = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
  
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  
  # optimizer = SGD(lr = 0.01, decay = 1E-9, momentum = 0.9, nesterov = True)
  # optimizer = Adam()
  # optimizer = RMSprop(lr = 0.01, rho = 0.9, epsilon = 1e-8, decay = 0.0)
  optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=0.0)
  
  model = Model(inputs = main_input, outputs = conv10)
  model.compile(optimizer = optimizer, loss = loss_jaccard)
  model.load_weights(args.model)
  model.summary()
  
  ##############################################################################
  # Train
  loss   = 0.
  losses = []
  epoch  = 0
  n_epochs = 200
  while epoch < n_epochs and loss > - float(args.loss):
    
    epoch += 1
    print ("Epoch: ", epoch, "of ", n_epochs)
    
    h = model.fit(x_data_filtered, y_data_filtered, shuffle = True, epochs = 1)
    # h = model.fit_generator(train_generator, steps_per_epoch = steps_per_epoch,
    #                         epochs = 1)
    
    loss = h.history['loss'][0]
    losses.append(loss)

################################################################################
# Predictions
if save_model(model, 'model_epoch' + str(epoch) + '_loss' + str(loss)): 
  print ('Model checkpoint saved')
  
predictions = model.predict(x_data_filtered)
for i in range(100):
  
  image = np.swapaxes(predictions[i], 0, 2)
  np.swapaxes(image, 1, 2)
  plot = plt.imshow(image[0])
  plt.colorbar()
  plt.savefig('img/test_out_' + str(i) + '.png')
  plt.close()
  
  image = np.swapaxes(y_data_filtered[i], 0, 2)
  plot = plt.imshow(image[0])
  plt.colorbar()
  plt.savefig('img/test_true_' + str(i) + '.png')
  plt.close()
  
  image = np.swapaxes(x_data_filtered[i], 0, 2)
  plot = plt.imshow(image[0])
  plt.colorbar()
  plt.savefig('img/test_raw_' + str(i) + '.png')
  plt.close()
  
plot = plt.plot(losses)
plt.savefig('img/losses.png')
plt.close()
