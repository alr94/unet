# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-c', '--config', help = 'JSON with script configuration')
parser.add_argument('-o', '--output', help = 'Output model file name')
parser.add_argument('-g', '--gpu',    help = 'Which GPU index', default = '0')
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
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

################################################################################
# My stuff
from utils import save_model, get_unet_data
from losses import *
from unet import *

################################################################################
# Get data, and filter
x_data, y_data = get_unet_data()
n_patches, patch_w, patch_h, patch_depth = x_data.shape

################################################################################ 
# Testing for loss functions, and filtering data
r = loss_jaccard(K.variable(y_data), 
                 K.variable(y_data)).eval(session = K.get_session())
 
idx = np.any(r < - 1e-10, axis = 1)
x_data_filtered = x_data[idx]
y_data_filtered = y_data[idx]
n_patches, patch_w, patch_h, patch_depth = x_data_filtered.shape


################################################################################
# Model definition 
sess = tf.InteractiveSession()
with sess.as_default():
  
  model = unet(inputshape = (patch_w, patch_h, patch_depth))
  
  loss, losses    =  0., []
  epoch, n_epochs =  0, 200
  batch_size      =  16
  
  while epoch < n_epochs and loss > -0.9:
    
    epoch += 1
    print ("Epoch: ", epoch, "of ", n_epochs)

    h = model.fit(x_data_filtered, y_data_filtered, batch_size = batch_size,
                  shuffle = True, epochs = 1)
    
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
