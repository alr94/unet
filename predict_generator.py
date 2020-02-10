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
  e_threshed = energy * mask
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
  locs = locs - [shape[0]/2, shape[1]/2, 0]
  return (np.linalg.norm(locs, axis=1))
  
  # for loc in locs:
  #   dist = math.sqrt((loc[0])**2 + (loc[1])**2)
  #   distances.push_back(dist)
  # return distances
    
    
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
                         dirname = 'MichelEnergyImage', 
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
  b_rq         = array('f', [0.])
  b_re         = array('f', [0.])
  # b_ret        = array('f', [0.])
  b_p_id       = array('f', [0.])
  b_p_vx       = array('f', [0.])
  b_p_vy       = array('f', [0.])
  b_p_vz       = array('f', [0.])
  b_p_ex       = array('f', [0.])
  b_p_ey       = array('f', [0.])
  b_p_ez       = array('f', [0.])
  b_p_ht       = array('f', [0.])
  b_p_t0       = array('f', [0.])
  b_d_nh       = array('f', [0.])
  b_d_nmh      = array('f', [0.])
  b_d_fmh      = array('f', [0.])
  b_d_dp       = array('f', [0.])
  b_d_dpx      = array('f', [0.])
  b_d_dpy      = array('f', [0.])
  b_d_dpz      = array('f', [0.])
  
  t_ev.Branch('thresh'               , b_thresh     , 'b_thresh/F')
  t_ev.Branch('inv_thresh'           , b_inv_thresh , 'b_inv_thresh/F')
  t_ev.Branch('n_hits_image'         , b_nhi        , 'b_nhi/F')
  t_ev.Branch('reco_charge'          , b_rq         , 'b_rq/F')
  t_ev.Branch('reco_energy'          , b_re         , 'b_re/F')
  # t_ev.Branch('reco_energy_threshed' , b_ret        , 'b_ret/F')
  t_ev.Branch("primID"               , b_p_id       , "b_p_id/F")
  t_ev.Branch("primVertexX"          , b_p_vx       , "b_p_vx/F")
  t_ev.Branch("primVertexY"          , b_p_vy       , "b_p_vy/F")
  t_ev.Branch("primVertexZ"          , b_p_vz       , "b_p_vz/F")
  t_ev.Branch("primEndX"             , b_p_ex       , "b_p_ex/F")
  t_ev.Branch("primEndY"             , b_p_ey       , "b_p_ey/F")
  t_ev.Branch("primEndZ"             , b_p_ez       , "b_p_ez/F")
  t_ev.Branch("primHasT0"            , b_p_ht       , "b_p_ht/F")
  t_ev.Branch("primHasT0"            , b_p_t0       , "b_p_t0/F")
  t_ev.Branch("daugNHits"            , b_d_nh       , "b_d_nh/F")
  t_ev.Branch("daugNMichHits"        , b_d_nmh      , "b_d_nmh/F")
  t_ev.Branch("daugFracMichHits"     , b_d_fmh      , "b_d_fmh/F")
  t_ev.Branch("daugDistPrim"         , b_d_dp       , "b_d_dp/F")
  t_ev.Branch("daugDistPrimX"        , b_d_dpx      , "b_d_dpx/F")
  t_ev.Branch("daugDistPrimY"        , b_d_dpy      , "b_d_dpy/F")
  t_ev.Branch("daugDistPrimZ"        , b_d_dpz      , "b_d_dpz/F")
  
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
  
  threshes = [0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4]
  hc_avgs, ec_avgs, hp_avgs, ep_avgs = [], [], [], []
  for thresh in threshes:
    
    print ('Thresh', thresh)
    
    b_thresh[0] = thresh
    
    for i in range(len(predictions)):
      # print (i , len(predictions))
      
      nHitImage = NHits(predictions[i], thresh)
      # hit_dists = HitDistances(predictions[i], thresh)
      
      recoQ         = RecoEnergy(predictions[i], test_charge[i], thresh)
      recoE         = RecoEnergy(predictions[i], test_energy[i], thresh)
      # recoEThreshed = RecoEnergyHitCut(predictions[i], test_energy[i], thresh)
      
      b_inv_thresh[0] = 1. / (1. - thresh)
      b_nhi[0]        = nHitImage
      b_rq[0]         = recoQ
      b_re[0]         = recoE
      # b_ret[0]        = recoEThreshed
      
      event = test_gen.keys[i].ReadObj()
      tree  = event.Get("param tree")
      for entry in tree:
        b_p_id[0]  = entry.primaryID
        b_p_vx[0]  = entry.VertexX
        b_p_vy[0]  = entry.VertexY
        b_p_vz[0]  = entry.VertexZ
        b_p_ex[0]  = entry.EndX
        b_p_ey[0]  = entry.EndY
        b_p_ez[0]  = entry.EndZ
        b_p_ht[0]  = entry.HasT0
        b_p_t0[0]  = entry.T0
        b_d_nh[0]  = entry.NHits
        b_d_nmh[0] = entry.NMichelHits
        b_d_fmh[0] = entry.FractionMichelHits
        b_d_dp[0]  = entry.DistanceToPrimary
        b_d_dpx[0] = entry.DistanceToPrimaryX
        b_d_dpy[0] = entry.DistanceToPrimaryY
        b_d_dpz[0] = entry.DistanceToPrimaryZ
        break
      ROOT.SetOwnership(tree, True)
      t_ev.Fill()
      
  print ('Reco tree filled')
  f.Write()
  f.Close()
  print ('Finished')
