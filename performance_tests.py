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
def HitCompleteness(pred, true, thresh):
  threshed  = (pred > thresh).astype(float)
  n_true    = np.sum(np.abs(true))
  n_correct = np.sum(np.abs(true * threshed))
  if n_true < 1e-10: return 0.
  completeness = n_correct / n_true
  return completeness

def EnergyCompleteness(pred, true, energy, thresh):
  threshed  = (pred > thresh).astype(float)
  e_true       = np.sum(np.abs(true * energy))
  e_correct    = np.sum(np.abs(threshed * true * energy))
  if e_true < 1e-10: return 0.
  completeness = e_correct / e_true
  return completeness
  
def HitPurity(pred, true, thresh):
  threshed  = (pred > thresh).astype(float)
  n_selected = np.sum(np.abs(threshed))
  n_correct  = np.sum(np.abs(true * threshed))
  if n_correct < 1e-10: return 0.
  purity = float(n_correct) / float(n_selected)
  return purity
  
def EnergyPurity(pred, true, energy, thresh):
  threshed  = (pred > thresh).astype(float)
  e_selected = np.sum(np.abs(threshed * energy))
  e_correct  = np.sum(np.abs(threshed * true * energy))
  if e_selected < 1e-10: return 0.
  purity = e_correct / e_selected
  return purity

def RecoEnergy(pred, energy, thresh):
  threshed  = (pred > thresh).astype(float)
  e_selected = np.sum(np.abs(threshed * energy))
  return e_selected
  
def TrueEnergy(true, energy):
  e_selected = np.sum(np.abs(true * energy))
  return e_selected

def NHits(pred, thresh):
  threshed  = (pred > thresh).astype(float)
  n_selected = np.sum(np.abs(threshed))
  return n_selected

################################################################################

if not (args.input and args.weights):
  print ('Please provide data, model, and weights')
  exit()
  
n_channels       = 3
conv_depth       = 3
patch_w, patch_h = 160, 160
batch_size       = 1

print ('Building data generator')
test_gen = DataGenerator(dataset_type = 'test', batch_size = batch_size, 
                         shuffle = False, root_data = args.input, 
                         patch_w = patch_w, patch_h = patch_h, 
                         patch_depth = n_channels)

# TODO: use true energy
  

sess = tf.InteractiveSession()
with sess.as_default():
  
  print ('Loading model')
  model     = unet(inputshape = (patch_w, patch_h, n_channels), 
                   conv_depth = conv_depth)
  model.load_weights(args.weights)
  
  print('Reformating data')
  test_x = np.zeros((test_gen.__len__(), patch_w, patch_h, 
    n_channels))
  test_y = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  for i in range(test_gen.__len__()):
    test_x[i], test_y[i] = test_gen.__getitem__(i)
    
  print ('Making predictions')
  predictions = model.predict_on_batch(test_x[:128])
  print ('Made predictions')
  print (predictions.shape, test_x[:128][..., 0].shape)
  flat = predictions.flatten()
  e_flat = test_x[:128][..., 0].flatten()
  print (flat.shape, e_flat.shape)
  
  
  ##############################################################################
  print ('Drawing score distribution')
  plot = plt.hist(flat, bins=100)
  plt.title('Hit Score Distribution')
  plt.yscale('log', nonposy='clip')
  plt.savefig('img/score_distribution.png')
  plt.close()
  
  plot = plt.hist(flat[flat > 0.8], bins='sqrt')
  plt.title('Hit Score Distribution')
  plt.yscale('log', nonposy='clip')
  plt.savefig('img/score_distribution_zoom.png')
  plt.close()
  ##############################################################################
  
  threshes = [0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1 - 1e-5, 
              1 - 1e-6, 1 - 1e-7]
  hc_avgs, ec_avgs, hp_avgs, ep_avgs  = [], [], [], []
  
  for thresh in threshes:
    
    ############################################################################
    print ('Drawing hit energy distribution')
    plot = plt.hist(e_flat[flat > thresh], bins='sqrt')
    plt.title('Hit Energy Distribution')
    plt.savefig('img/hit_energy_distribution_thresh' + str(thresh) + '.png')
    plt.close()
    ############################################################################
    
    print ('Evaluating performance metrics at threshold' + str(thresh))
    hc, ec         = [0.] * len(predictions), [0.] * len(predictions)
    hp, ep         = [0.] * len(predictions), [0.] * len(predictions)
    trueEs, recoEs = [], []
    normDiffs      = []
    nHits          = []
    
    for i in range(len(predictions)):
      
      trueE = TrueEnergy(test_y[i], test_x[i][..., 0])
      recoE = RecoEnergy(predictions[i], test_x[i][..., 0], thresh)
      normDiff = (recoE - trueE) / trueE
      trueEs.append(trueE)
      recoEs.append(recoE)
      if normDiff == normDiff: normDiffs.append(normDiff)
      
      nHit = NHits(predictions[i], thresh)
      nHits.append(nHit)
      
      if i % 100 == 0: print (i)
      hc[i] = HitCompleteness(predictions[i], test_y[i], thresh)
      ec[i] = EnergyCompleteness(predictions[i], test_y[i], test_x[i][...,0], 
                                 thresh)
      hp[i] = HitPurity(predictions[i], test_y[i], thresh)
      ep[i] = EnergyPurity(predictions[i], test_y[i], test_x[i][...,0], thresh)
        
    hc = [x for x in hc if x > 1e-10]
    ec = [x for x in ec if x > 1e-10]
    hp = [x for x in hp if x > 1e-10]
    ep = [x for x in ep if x > 1e-10]
    
    hc_avg, ec_avg, hp_avg, ep_avg = 0., 0., 0., 0.
    for i in range(len(hc)): 
      hc_avg += hc[i]
      ec_avg += ec[i]
      hp_avg += hp[i]
      ep_avg += ep[i]
    hc_avg /= len(hc)
    ec_avg /= len(ec)
    hp_avg /= len(hp)
    ep_avg /= len(ep)
    
    hc_avgs.append(hc_avg)
    ec_avgs.append(ec_avg)
    hp_avgs.append(hp_avg)
    ep_avgs.append(ep_avg)
      
    plot = plt.hist(hc, bins='sqrt')
    plt.title('Hit Completeness: Threshold ' + str(thresh))
    plt.savefig('img/hit_comp_thresh' + str(thresh) + '_avg' + str(hc_avg) + '.png')
    plt.close()
     
    plot = plt.hist(ec, bins='sqrt')
    plt.title('Energy Completeness: Threshold ' + str(thresh))
    plt.savefig('img/energy_comp_thresh' + str(thresh) + '_avg' + str(ec_avg) + '.png')
    plt.close()
     
    plot = plt.hist(hp, bins='sqrt')
    plt.title('Hit Purity: Threshold ' + str(thresh))
    plt.savefig('img/hit_pur_thresh' + str(thresh) + '_avg' + str(hp_avg) + '.png')
    plt.close()
     
    plot = plt.hist(ep, bins='sqrt')
    plt.title('Energy Purity: Threshold ' + str(thresh))
    plt.savefig('img/energy_pur_thresh' + str(thresh) + '_avg' + str(ep_avg) + '.png')
    plt.close()
    
    plot = plt.hist(trueEs, bins='sqrt')
    plt.title('True Visible Energy')
    plt.xlabel('Integrated Hit ADC')
    plt.savefig('img/true_visible.png')
    plt.close()
    
    plot = plt.hist(recoEs, bins='sqrt')
    plt.title('Reco Energy')
    plt.xlabel('Integrated Hit ADC')
    plt.savefig('img/reco_energy_thresh' + str(thresh) + '.png')
    plt.close()
    
    ND_mean   = np.mean(normDiffs)
    ND_stddev = np.std(normDiffs)
    print (ND_mean, ND_stddev)
    plot = plt.hist(normDiffs, bins='sqrt')
    plt.title('Normalised energy Difference')
    plt.xlabel('(reco - true) / true')
    plt.savefig('img/norm_diff_thresh' + str(thresh) + '.png')
    plt.close()
    
    plot = plt.hist(nHits, bins='sqrt')
    plt.title('Number Selected Hits')
    plt.xlabel('N Hits')
    plt.savefig('img/nhits_thresh' + str(thresh) + '.png')
    plt.close()
  
  plot = plt.plot(hc_avgs, hp_avgs, '.-')
  plt.title('Hit Completeness vs Hit Purity')
  plt.xlabel('Hit Completeness')
  plt.xlabel('Hit Purity')
  plt.savefig('img/hit_comp_v_pur.png')
  plt.close()
  
  plot = plt.plot(ec_avgs, ep_avgs, '.-')
  plt.title('Energy Completeness vs Energy Purity')
  plt.xlabel('Energy Completeness')
  plt.xlabel('Energy Purity')
  plt.savefig('img/energy_comp_v_pur.png')
  plt.close()
      
      # if i < 100:
      #   image = np.swapaxes(predictions[i], 0, 2)
      #   np.swapaxes(image, 1, 2)
      #   plot = plt.imshow(image[0])
      #   plt.colorbar()
      #   plt.savefig('img/out_' + str(i) + '_hc' + str(hc[i]) + '_hp' + str(hp[i]) 
      #               + '.png')
      #   plt.close()
      #   
      #   image = np.swapaxes(test_y[i], 0, 2)
      #   plot = plt.imshow(image[0])
      #   plt.colorbar()
      #   plt.savefig('img/true_' + str(i) + '_hc' + str(hc[i]) + '_hp' + str(hp[i]) 
      #               + '.png')
      #   plt.close()
      #   
      #   image = np.swapaxes(test_x[i], 0, 2)
      #   plot = plt.imshow(image[0])
      #   plt.colorbar()
      #   plt.savefig('img/raw_' + str(i) + '_hc' + str(hc[i]) + '_hp' + str(hp[i]) 
      #               + '.png')
      #   plt.close()
