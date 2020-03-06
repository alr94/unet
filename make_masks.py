# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-i', '--input',   help = 'Input file')
parser.add_argument('-o', '--output',   help = 'Output file')
parser.add_argument('-w', '--weights', help = 'Weights file (optional)')
args = parser.parse_args()

################################################################################
# setup tensorflow enviroment variables
import os
from os.path import exists, isfile, join
os.environ['KERAS_BACKEND']        = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
#keras.backend.set_image_dim_ordering('tf')
        
################################################################################
# Other setups
import numpy as np
import pandas as pd

################################################################################
# My stuff
from losses import *
from unet import *
from data_gen import DataGenerator
################################################################################

if not (args.input and args.weights):
  print ('Please provide data, model, and weights')
  exit()
  
n_channels       = 3
conv_depth       = 3
patch_w, patch_h = 160, 160
batch_size       = 64
steps            = 0

print ('Building data generator')
test_gen = DataGenerator(dataset_type = 'data', 
                         dirname = 'MichelEnergyImageData', 
                         batch_size = batch_size, shuffle = False, 
                         root_data = args.input, patch_w = patch_w, 
                         patch_h = patch_h, patch_depth = n_channels)

sess = tf.InteractiveSession()
with sess.as_default():
  
  print ('Loading model')
  model     = unet(inputshape = (patch_w, patch_h, n_channels), 
                   conv_depth = conv_depth)
  model.load_weights(args.weights)
  
  print ('Loading charge info')
  if steps == 0:
    test_charge = np.zeros((test_gen.__len__() * batch_size,  patch_w, patch_h, 1))
    test_energy = np.zeros((test_gen.__len__() * batch_size,  patch_w, patch_h, 1))
    for i in range(test_gen.__len__()):
      wires    = test_gen.getitembykey(i, 'wire')
      energies = test_gen.getitembykey(i, 'energy')
      for j in range(batch_size):
        test_charge[(i*batch_size) + j] = wires[j]
        test_energy[(i*batch_size) + j] = energies[j] 
  else:
    test_charge = np.zeros((steps * batch_size,  patch_w, patch_h, 1))
    test_energy = np.zeros((steps * batch_size,  patch_w, patch_h, 1))
    for i in range(steps):
      wires    = test_gen.getitembykey(i, 'wire')
      energies = test_gen.getitembykey(i, 'energy')
      for j in range(batch_size):
        test_charge[(i*batch_size) + j] = wires[j]
        test_energy[(i*batch_size) + j] = energies[j] 
      
     
  print ('Making predictions')
  if steps == 0: 
    predictions = model.predict_generator(test_gen, verbose = 1)
  else: 
    predictions = model.predict_generator(test_gen, verbose = 1, steps = steps)
  print ('Made predictions')
  
  np.savez(args.output, test_charge, test_energy, predictions)
