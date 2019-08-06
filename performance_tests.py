# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-i', '--input',   help = 'Input directory')
parser.add_argument('-w', '--weights', help = 'Weights file (optional)')

args = parser.parse_args()

################################################################################
# setup tensorflow enviroment variables
import os
from os.path import exists, isfile, join
os.environ['KERAS_BACKEND']        = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
import math
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

################################################################################
# My stuff
from losses import *
from unet import *
from data_gen import DataGenerator

################################################################################
# My metrics
def HitCompleteness(pred, true):
  n_true       = np.sum(np.abs(true))
  n_correct    = np.sum(np.abs(true * pred))
  if n_true < 1e-10: return 0.
  completeness = n_correct / n_true
  return completeness

def EnergyCompleteness(pred, true, energy):
  e_true       = np.sum(np.abs(true * energy))
  e_correct    = np.sum(np.abs(pred * true * energy))
  if e_true < 1e-10: return 0.
  completeness = e_correct / e_true
  return completeness
  
def HitPurity(pred, true):
  n_selected = np.sum(np.abs(pred))
  n_correct  = np.sum(np.abs(true * pred))
  if n_correct < 1e-10: return 0.
  purity = float(n_correct) / float(n_selected)
  return purity
  
def EnergyPurity(pred, true, energy):
  e_selected = np.sum(np.abs(pred * energy))
  e_correct  = np.sum(np.abs(pred * true * energy))
  if e_selected < 1e-10: return 0.
  purity = e_correct / e_selected
  return purity

################################################################################

if not (args.input and args.weights):
  print ('Please provide data, model, and weights')
  exit()
  
n_channels       = 3
patch_w, patch_h = 160, 160
batch_size = 1

print ('Building data generator')
test_gen = DataGenerator(dataset_type = 'test', batch_size = batch_size, 
                         shuffle = False, root_data = args.input, 
                         patch_w = patch_w, patch_h = patch_h, 
                         patch_depth = n_channels)

# TODO: use true energy

sess = tf.InteractiveSession()
with sess.as_default():
  
  print ('Loading model')
  model     = unet(inputshape = (patch_w, patch_h, n_channels), conv_depth = 3)
  optimizer = Nadam()
  
  model.compile(optimizer = optimizer, loss = jaccard_distance)
  model.summary()
  model.load_weights(args.weights)
  
  print ('Evaluating model')
  score = model.evaluate_generator(test_gen,verbose = True)
  print (score)
  
  print('Reformating data')
  test_x = np.zeros((test_gen.__len__(), patch_w, patch_h, 
    n_channels))
  test_y = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  for i in range(test_gen.__len__()):
    test_x[i], test_y[i] = test_gen.__getitem__(i)
    
  print ('Making predictions')
  predictions = model.predict(test_x)
  
  print ('Evaluating performance metrics')
  hc = [0.] * len(predictions)
  ec = [0.] * len(predictions)
  hp = [0.] * len(predictions)
  ep = [0.] * len(predictions)
  
  for i in range(len(predictions)):
    
    if i % 100 == 0: print (i)
    hc[i] = HitCompleteness(predictions[i], test_y[i])
    ec[i] = EnergyCompleteness(predictions[i], test_y[i], test_x[i][...,0])
    hp[i] = HitPurity(predictions[i], test_y[i])
    ep[i] = EnergyPurity(predictions[i], test_y[i], test_x[i][...,0])
    
    if i < 100:
      image = np.swapaxes(predictions[i], 0, 2)
      np.swapaxes(image, 1, 2)
      plot = plt.imshow(image[0])
      plt.colorbar()
      plt.savefig('img/out_' + str(i) + '_hc' + str(hc[i]) + '_hp' + str(hp[i]) 
                  + '.png')
      plt.close()
      
      image = np.swapaxes(test_y[i], 0, 2)
      plot = plt.imshow(image[0])
      plt.colorbar()
      plt.savefig('img/true_' + str(i) + '_hc' + str(hc[i]) + '_hp' + str(hp[i]) 
                  + '.png')
      plt.close()
      
      image = np.swapaxes(test_x[i], 0, 2)
      plot = plt.imshow(image[0])
      plt.colorbar()
      plt.savefig('img/raw_' + str(i) + '_hc' + str(hc[i]) + '_hp' + str(hp[i]) 
                  + '.png')
      plt.close()
      
  hc = [x for x in hc if x > 1e-10]
  ec = [x for x in ec if x > 1e-10]
  hp = [x for x in hp if x > 1e-10]
  ep = [x for x in ep if x > 1e-10]
  
  hc_avg, ec_avg, hp_avg, ep_avg = 0., 0., 0., 0.
  for i in range(len(hc)): hc_avg += hc[i]
  for i in range(len(ec)): ec_avg += ec[i]
  for i in range(len(hp)): hp_avg += hp[i]
  for i in range(len(ep)): ep_avg += ep[i]
  hc_avg /= len(hc)
  ec_avg /= len(ec)
  hp_avg /= len(hp)
  ep_avg /= len(ep)
  
  print (hc_avg)
  print (ec_avg)
  print (hp_avg)
  print (ep_avg)
    
  print ('Making plots')

  plot = plt.hist(hc, bins='sqrt')
  plt.savefig('img/hit_comp_avg' + str(hc_avg) + '.png')
  plt.close()
   
  plot = plt.hist(ec, bins='sqrt')
  plt.savefig('img/energy_comp_avg' + str(ec_avg) + '.png')
  plt.close()
   
  plot = plt.hist(hp, bins='sqrt')
  plt.savefig('img/hit_pur._avg' + str(hp_avg) + '.png')
  plt.close()
   
  plot = plt.hist(ep, bins='sqrt')
  plt.savefig('img/energy_pur._avg' + str(ep_avg) + '.png')
  plt.close()
