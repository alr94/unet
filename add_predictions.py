# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-i', '--input',   help = 'Input directory')
parser.add_argument('-o', '--output',  help = 'Output directory', default = 'test.root')
parser.add_argument('-w', '--weights', help = 'Weights file (optional)')
args = parser.parse_args()

################################################################################
# setup tensorflow enviroment variables
import os
from os.path import exists, isfile, join
os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
import math
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
  
import ROOT
from ROOT import TFile, TTree

import root_numpy

from array import array

################################################################################
# My stuff
from losses import *
from unet import *
from data_gen import DataGenerator

################################################################################
if not (args.input and args.weights):
  print ('Please provide data, model, and weights')
  exit()

fname = args.input.split('Image')[0] + 'Prediction' + args.input.split('Image')[1]
print (os.path.basename(fname))
if os.path.basename(fname) in os.listdir(os.path.dirname(args.input)): 
  print ('Already predicted this file, skipping.')
  exit()
  
################################################################################
dataset_type     = 'data'
dirname          = 'MichelEnergyImageData' if 'Data' in args.input else 'MichelEnergyImage'
n_channels       = 1
conv_depth       = 2
patch_w, patch_h = 160, 160
batch_size       = 16
steps            = 1

################################################################################
print ('Building data generator')

test_gen = DataGenerator(dataset_type = dataset_type, 
                         dirname = dirname, 
                         batch_size = batch_size, shuffle = False, 
                         root_data = args.input, patch_w = patch_w, 
                         patch_h = patch_h, patch_depth = n_channels)

################################################################################
sess = tf.InteractiveSession()
with sess.as_default():
  
  print ('Loading model')
  if 'inception' in args.weights:
    model     = inception_unet(inputshape = (patch_w, patch_h, n_channels), 
                               conv_depth = conv_depth,
                               number_base_nodes = int(args.weights.split('_')[2][:2]))
  else:
    model     = unet(inputshape = (patch_w, patch_h, n_channels), 
                               conv_depth = conv_depth)
    
  model.load_weights(args.weights)
  
  if steps == 0: 
    predictions = model.predict_generator(test_gen, verbose = 1)
  else: 
    predictions = model.predict_generator(test_gen, verbose = 1, steps = steps)
  
  print ('Saving to ' + fname)
  
  f = ROOT.TFile(fname, 'RECREATE')
  for i, key in enumerate(test_gen.keys):
    
    if i >= len(predictions): break
    # if i >= 1: break
    
    # Make the new hist
    hist = ROOT.TH2D('prediction', 'prediction', 
                     patch_w, 0, patch_w, patch_h, 0, patch_h)
    
    # Get the predicted array and add overflow bins
    arr = np.zeros((predictions[i].shape[0] + 2, predictions[i].shape[1] + 2), 
                   dtype = 'float32')
    arr[1:1 + predictions[i].shape[0], 
        1:1 + predictions[i].shape[1]] = predictions[i][:,:,0]
    
    arr[arr < 0.001] = 0.
    
    # Fill hist and write
    root_numpy.array2hist(arr, hist)
    
    # Write hist to TDirectoryFile
    eventdir = key.GetTitle()
    f.mkdir(dirname + '/' + eventdir)
    f.cd(dirname + '/' + eventdir)
    hist.Write() 
    
print ('Done')
