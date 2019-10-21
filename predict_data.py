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
  
from ROOT import TFile, TTree
from array import array

################################################################################
# My stuff
from losses import *
from unet import *
from data_gen import DataGenerator

################################################################################
# My metrics
def RecoEnergy(pred, energy, thresh):
  threshed   = (pred > thresh).astype(float)
  e_selected = np.sum(np.abs(threshed * energy))
  return e_selected
  
def NHits(pred, thresh):
  threshed   = (pred > thresh).astype(float)
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
# FIXME
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
  
  print('Reformating data')
  test_x      = np.zeros((test_gen.__len__(), patch_w, patch_h, n_channels))
  test_y      = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  test_charge = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  test_energy = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  for i in range(test_gen.__len__()):
    test_x[i], test_y[i] = test_gen.__getitem__(i)
    test_charge[i]       = test_gen.getitembykey(i, 'wire')
    test_energy[i]       = test_gen.getitembykey(i, 'energy')
     
  print ('Making predictions')
  predictions = model.predict(test_x, batch_size = 8, verbose = 1)
  del test_x, test_y
    
  print ('Made predictions')
  q_flat = test_charge[..., 0].flatten()
  flat   = predictions.flatten()[q_flat > 0.1]
  e_flat = test_energy[..., 0].flatten()[q_flat > 0.1]
  q_flat = q_flat[q_flat > 0.1]
  
  print (len(flat), len(q_flat))
  
  f = TFile('predict_output.root', 'recreate')
  
  ##############################################################################
  t_ev = TTree('performance_tests', 'Thresholded performance scores')
  b_thresh     = array('f', [0.])
  b_inv_thresh = array('f', [0.])
  b_nhi        = array('f', [0.])
  b_rq         = array('f', [0.])
  b_re         = array('f', [0.])
  t_ev.Branch('thresh'       , b_thresh     , 'b_thresh/F')
  t_ev.Branch('inv_thresh'   , b_inv_thresh , 'b_inv_thresh/F')
  t_ev.Branch('n_hits_image' , b_nhi        , 'b_nhi/F')
  t_ev.Branch('reco_charge'  , b_rq         , 'b_rq/F')
  t_ev.Branch('reco_energy'  , b_re         , 'b_re/F')
  ##############################################################################
  t_hit = TTree('hit_metrics', 'Thresholded hit scores')
  b_hit_cnn    = array('f', [0.])
  b_hit_int    = array('f', [0.])
  b_hit_energy = array('f', [0.])
  t_hit.Branch('hit_cnn'    , b_hit_cnn    , 'b_hit_cnn/F')
  t_hit.Branch('hit_int'    , b_hit_int    , 'b_hit_int/F')
  t_hit.Branch('hit_energy' , b_hit_energy , 'b_hit_energy/F')
  ##############################################################################
  
  print ('Trees made')
  for i in range(len(flat)):
    if i % 1000 == 0: print(i, len(flat))
    b_hit_cnn[0]    = flat[i]
    b_hit_int[0]    = q_flat[i]
    b_hit_energy[0] = e_flat[i]
    t_hit.Fill()
  print ('Hit Tree filled')
  
  threshes = [0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1 - 1e-5, 
              1 - 1e-6, 1 - 1e-7]
  hc_avgs, ec_avgs, hp_avgs, ep_avgs = [], [], [], []
  
  for thresh in threshes:
    
    print ('Thresh', thresh)
    
    b_thresh[0] = thresh
      
    integratedQs, recoQs = [], []
    integratedEs, recoEs = [], []
    nHits                = []
    
    for i in range(len(predictions)):
      
      nHitImage  = NHits(predictions[i], thresh)
      
      recoQ       = RecoEnergy(predictions[i], test_charge[i], thresh)
      recoE       = RecoEnergy(predictions[i], test_energy[i], thresh)
      
      b_inv_thresh[0] = 1. / (1. - thresh)
      b_nhi[0]        = nHitImage
      b_rq[0]         = recoQ
      b_re[0]         = recoE
      
      t_ev.Fill()
      
      recoQs.append(recoQ)
      recoEs.append(recoE)
      nHits.append(nHitImage)
      
  print ('Reco tree filled')
        
  f.Write()
  f.Close()
