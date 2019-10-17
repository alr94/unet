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

################################################################################
# My stuff
from losses import *
from unet import *
from data_gen import DataGenerator

################################################################################
# My metrics
def HitCompleteness(pred, true, ntrue, thresh):
  threshed  = (pred > thresh).astype(float)
  # n_true    = np.sum(np.abs(true))
  n_correct = np.sum(np.abs(true * threshed))
  if ntrue < 1e-10: 
    if n_correct < 1e-10: return 1.
    else: return 0.
  completeness = n_correct / ntrue
  return completeness

def EnergyCompleteness(pred, true, energy, thresh):
  threshed  = (pred > thresh).astype(float)
  e_true    = np.sum(np.abs(true * energy))
  e_correct = np.sum(np.abs(threshed * true * energy))
  if e_true < 1e-10: 
    if e_correct < 1e-10: return 1.
    else: return 0.
  completeness = e_correct / e_true
  return completeness
  
def HitPurity(pred, true, thresh):
  threshed   = (pred > thresh).astype(float)
  n_selected = np.sum(np.abs(threshed))
  n_correct  = np.sum(np.abs(true * threshed))
  if n_correct < 1e-10: return 0.
  purity = float(n_correct) / float(n_selected)
  return purity
  
def EnergyPurity(pred, true, energy, thresh):
  threshed   = (pred > thresh).astype(float)
  e_selected = np.sum(np.abs(threshed * energy))
  e_correct  = np.sum(np.abs(threshed * true * energy))
  if e_selected < 1e-10: return 0.
  purity = e_correct / e_selected
  return purity

def RecoEnergy(pred, energy, thresh):
  threshed   = (pred > thresh).astype(float)
  e_selected = np.sum(np.abs(threshed * energy))
  return e_selected
  
def TrueEnergy(true, energy):
  e_selected = np.sum(np.abs(true * energy))
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
test_gen = DataGenerator(dataset_type = 'all', dirname = 'MichelEnergyImage', 
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
  test_true   = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  test_charge = np.zeros((test_gen.__len__(), patch_w, patch_h, 1))
  test_e      = np.zeros((test_gen.__len__(), 1))
  test_n      = np.zeros((test_gen.__len__(), 1))
  for i in range(test_gen.__len__()):
    test_x[i], test_y[i] = test_gen.__getitem__(i)
    test_true[i]         = test_gen.getitembykey(i, 'trueEnergy')
    test_charge[i]       = test_gen.getitembykey(i, 'energy')
    test_e[i]            = test_gen.getenergy(i)
    test_n[i]            = test_gen.getitembykey(i, 'nTrue')[0,0,0]
    
  #FIXME
  # test_x    = test_x[:8]
  # test_y    = test_y[:8]
  # test_true = test_true[:8]
  # test_e    = test_e[:8]
    
  print ('Making predictions')
  predictions = model.predict(test_x, batch_size = 8, verbose = 1)
  print ('Made predictions')
  flat = predictions.flatten()
  true_flat = test_y.flatten()
  e_flat = test_charge[..., 0].flatten()
  
  from ROOT import TFile, TTree
  from array import array
  
  # FIXME
  # f = TFile('performance_' + args.weights.split('/')[-2] + '.root', 'recreate')
  f = TFile('test.root', 'recreate')
  
  t_ev = TTree('performance_tests', 'Thresholded performance scores')
  b_thresh     = array('f', [0.])
  b_inv_thresh = array('f', [0.])
  b_energy     = array('f', [0.])
  b_hc         = array('f', [0.])
  b_ec         = array('f', [0.])
  b_hp         = array('f', [0.])
  b_ep         = array('f', [0.])
  b_ndq        = array('f', [0.])
  b_nde        = array('f', [0.])
  b_nt         = array('f', [0.])
  b_nti        = array('f', [0.])
  b_nhi        = array('f', [0.])
  b_iq         = array('f', [0.])
  b_rq         = array('f', [0.])
  b_ie         = array('f', [0.])
  b_re         = array('f', [0.])
  t_ev.Branch('thresh'              , b_thresh     , 'b_thresh/F')
  t_ev.Branch('inv_thresh'          , b_inv_thresh , 'b_inv_thresh/F')
  t_ev.Branch('energy'              , b_energy     , 'b_energy/F')
  t_ev.Branch('hit_completeness'    , b_hc         , 'b_hc/F')
  t_ev.Branch('energy_completeness' , b_ec         , 'b_ec/F')
  t_ev.Branch('hit_purity'          , b_hp         , 'b_hp/F')
  t_ev.Branch('energy_purity'       , b_ep         , 'b_ep/F')
  t_ev.Branch('norm_diff_charge'    , b_ndq        , 'b_ndq/F')
  t_ev.Branch('norm_diff_energy'    , b_nde        , 'b_nde/F')
  t_ev.Branch('n_true'              , b_nt         , 'b_nt/F')
  t_ev.Branch('n_trueimage'         , b_nti        , 'b_nti/F')
  t_ev.Branch('n_hits_image'        , b_nhi        , 'b_nhi/F')
  t_ev.Branch('integrated_charge'   , b_iq         , 'b_iq/F')
  t_ev.Branch('reco_charge'         , b_rq         , 'b_rq/F')
  t_ev.Branch('integrated_energy'   , b_ie         , 'b_ie/F')
  t_ev.Branch('reco_energy'         , b_re         , 'b_re/F')
  
  
  t_hit      = TTree('hit_metrics', 'Thresholded hit scores')
  b_hit_cnn  = array('f', [0.])
  b_hit_int  = array('f', [0.])
  b_hit_true = array('f', [0.])
  t_hit.Branch('hit_cnn' , b_hit_cnn , 'b_hit_cnn/F')
  t_hit.Branch('hit_int' , b_hit_int , 'b_hit_int/F')
  t_hit.Branch('hit_true' , b_hit_true , 'b_hit_true/F')
  ##############################################################################
    
  for i in range(len(flat)):
    b_hit_cnn[0]  = flat[i]
    b_hit_int[0]  = e_flat[i]
    b_hit_true[0] = true_flat[i]
    t_hit.Fill()
  
  threshes = [0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1 - 1e-5, 
              1 - 1e-6, 1 - 1e-7]
  hc_avgs, ec_avgs, hp_avgs, ep_avgs = [], [], [], []
  
  for thresh in threshes:
    
    b_thresh[0] = thresh
      
    print ('Evaluating performance metrics at threshold' + str(thresh))
    hcs, ecs             = [0.] * len(predictions), [0.] * len(predictions)
    hps, eps             = [0.] * len(predictions), [0.] * len(predictions)
    integratedQs, recoQs = [], []
    integratedEs, recoEs = [], []
    normDiffQs           = []
    normDiffEs           = []
    nHits                = []
    
    for i in range(len(predictions)):
      if i % 100 == 0: print (i)
      
      nTrue      = test_n[i][0]
      nTrueImage = np.sum(np.abs(test_y[i]))
      energy     = test_e[i][0]
      nHitImage  = NHits(predictions[i], thresh)
      
      integratedQ = TrueEnergy(test_y[i], test_charge[i])
      recoQ       = RecoEnergy(predictions[i], test_charge[i], thresh)
      integratedE = TrueEnergy(test_y[i], test_true[i])
      recoE       = RecoEnergy(predictions[i], test_true[i], thresh)
      
      normDiffQ = (recoQ - integratedQ) / integratedQ
      normDiffE = (recoE - integratedE) / integratedE
      
      hc   = HitCompleteness(predictions[i], test_y[i], nTrue, thresh)
      ec   = EnergyCompleteness(predictions[i], test_y[i], test_true[i], thresh)
      hp   = HitPurity(predictions[i], test_y[i], thresh)
      ep   = EnergyPurity(predictions[i], test_y[i], test_true[i], thresh)
      
      if not (normDiffQ == normDiffQ and normDiffQ != float('inf')): continue 
      if not (normDiffE == normDiffE and normDiffE != float('inf')): continue 
      
      b_inv_thresh[0] = 1. / (1. - thresh)
      b_energy[0]     = energy
      b_hc[0]         = hc
      b_ec[0]         = ec
      b_hp[0]         = hp
      b_ep[0]         = ep
      b_ndq[0]        = normDiffQ
      b_nde[0]        = normDiffE
      b_nt[0]         = nTrue
      b_nti[0]        = nTrueImage
      b_nhi[0]        = nHitImage
      b_iq[0]         = integratedQ
      b_rq[0]         = recoQ
      b_ie[0]         = integratedE
      b_re[0]         = recoE
      
      t_ev.Fill()
      
      integratedQs.append(integratedQ)
      recoQs.append(recoQ)
      integratedEs.append(integratedE)
      recoEs.append(recoE)
      normDiffQs.append(normDiffQ)
      normDiffEs.append(normDiffE)
      nHits.append(nHitImage)
      hcs.append(hc)
      ecs.append(ec)
      hps.append(hp)
      eps.append(ep)
        
    hcs = [x for x in hcs if x > 1e-10]
    ecs = [x for x in ecs if x > 1e-10]
    hps = [x for x in hps if x > 1e-10]
    eps = [x for x in eps if x > 1e-10]
     
    hc_avg, ec_avg, hp_avg, ep_avg = 0., 0., 0., 0.
    for i in range(len(hcs)): hc_avg += hcs[i]
    for i in range(len(ecs)): ec_avg += ecs[i]
    for i in range(len(hps)): hp_avg += hps[i]
    for i in range(len(eps)): ep_avg += eps[i]
    hc_avg /= len(hcs)
    ec_avg /= len(ecs)
    hp_avg /= len(hps)
    ep_avg /= len(eps)
     
    hc_avgs.append(hc_avg)
    ec_avgs.append(ec_avg)
    hp_avgs.append(hp_avg)
    ep_avgs.append(ep_avg)
    
  f.Write()
  f.Close()
  
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
