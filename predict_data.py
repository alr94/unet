# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-i', '--input',   help = 'Input directory')
parser.add_argument('-o', '--output',  help = 'Output directory')
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
  
import ROOT
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

def RecoEnergyHitCut(pred, energy, thresh):
  threshed   = (pred > thresh).astype(float)
  mask       = (energy > 0.75).astype(float)
  e_threshed = energy*mask
  e_selected = np.sum(np.abs(threshed * e_threshed))
  return e_selected
  
def NHits(pred, thresh):
  threshed   = (pred > thresh).astype(float)
  n_selected = np.sum(np.abs(threshed))
  return n_selected

def Locations(pred, thresh):
  locs = np.argwhere(pred > thresh)
  return locs

def HitDistances(pred, thresh):
  
  distances = ROOT.vector('float')()
  shape     = pred.shape
  
  locs = Locations(pred, thresh)
  
  for loc in locs:
    dist = math.sqrt((loc[0] - (shape[0]/2))**2 + (loc[1] - (shape[1]/2))**2)
    distances.push_back(dist)
    
  return distances

################################################################################

if not (args.input and args.weights):
  print ('Please provide data, model, and weights')
  exit()
  
n_channels       = 3
conv_depth       = 3
patch_w, patch_h = 160, 160
batch_size       = 1

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
  
  print('Reformating data')
  test_x      = np.zeros((test_gen.__len__(), patch_w, patch_h, n_channels))
  test_y      = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  test_charge = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  test_energy = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  for i in range(test_gen.__len__()):
    test_x[i], test_y[i] = test_gen.__getitem__(i)
    test_charge[i]       = test_gen.getitembykey(i, 'wire')
    test_energy[i]       = test_gen.getitembykey(i, 'energy')
  
  # FIXME
  # test_x      = test_x[:8] 
  # test_y      = test_y[:8]
  # test_charge = test_charge[:8]
  # test_energy = test_energy[:8] 
     
  print ('Making predictions')
  predictions = model.predict(test_x, batch_size = 8, verbose = 1)
  del test_x, test_y
    
  print ('Made predictions')
  q_flat = test_charge[..., 0].flatten()
  flat   = predictions.flatten()[q_flat > 0.1]
  e_flat = test_energy[..., 0].flatten()[q_flat > 0.1]
  q_flat = q_flat[q_flat > 0.1]
  
  f = TFile(args.output, 'recreate')
  
  ##############################################################################
  t_ev = TTree('performance_tests', 'Thresholded performance scores')
  b_thresh     = array('f', [0.])
  b_inv_thresh = array('f', [0.])
  b_nhi        = array('f', [0.])
  b_hd         = ROOT.vector('float')()
  b_rq         = array('f', [0.])
  b_re         = array('f', [0.])
  b_ret        = array('f', [0.])
  t_ev.Branch('thresh'        , b_thresh     , 'b_thresh/F')
  t_ev.Branch('inv_thresh'    , b_inv_thresh , 'b_inv_thresh/F')
  t_ev.Branch('n_hits_image'  , b_nhi        , 'b_nhi/F')
  t_ev.Branch('hit_distances' , b_hd)
  t_ev.Branch('reco_charge'   , b_rq         , 'b_rq/F')
  t_ev.Branch('reco_energy'   , b_re         , 'b_re/F')
  t_ev.Branch('reco_energy_threshed'   , b_ret         , 'b_ret/F')
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
    
    for i in range(len(predictions)):
      
      nHitImage  = NHits(predictions[i], thresh)
      
      hit_dists = HitDistances(predictions[i], thresh)
      
      recoQ         = RecoEnergy(predictions[i], test_charge[i], thresh)
      recoE         = RecoEnergy(predictions[i], test_energy[i], thresh)
      recoEThreshed = RecoEnergyHitCut(predictions[i], test_energy[i], thresh)
      
      b_inv_thresh[0] = 1. / (1. - thresh)
      b_nhi[0]        = nHitImage
      b_hd            = hit_dists
      b_rq[0]         = recoQ
      b_re[0]         = recoE
      b_ret[0]        = recoEThreshed
      
      t_ev.Fill()
      
  print ('Reco tree filled')
        
  f.Write()
  f.Close()
