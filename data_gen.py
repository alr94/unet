# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
import numpy as np
import keras

import ROOT
import root_numpy

from collections import defaultdict
import itertools
import time

from losses import *

class DataGenerator(keras.utils.Sequence):

  def __init__(self, root_data, dataset_type, dirname, batch_size = 32, 
               shuffle = True, patch_w = 160, patch_h = 160, patch_depth = 3, 
               val_frac = 0.05, test_frac = 0.05, number_keys = 0):
    
    self.dataset_type = dataset_type
    self.val_frac  = val_frac
    self.test_frac = test_frac
    
    self.batch_size   = batch_size
    self.shuffle      = shuffle
    
    self.dim         = (patch_w, patch_h, patch_depth)
    self.patch_w     = patch_w
    self.patch_h     = patch_h
    self.patch_depth = patch_depth
    
    self.n_crop = int((320 - patch_w)/ 2)
    
    self.TFile  = ROOT.TFile(root_data)
    self.TDir   = self.TFile.Get(dirname)
    self.TTree  = self.TDir.Get('param tree')
    
    self.data_types = []
    event = self.TDir.GetListOfKeys()[0].ReadObj()
    for evKey in event.GetListOfKeys():
      if evKey.GetClassName() == "TTree": continue
      keyType = evKey.GetName()
      if keyType not in self.data_types : self.data_types.append(keyType)
        
    # Previous method before changing the folder structure in the root files
    #self.keys     = defaultdict()
    #for data_type in self.data_types:
    #  self.keys[data_type] = [k for k in self.TDir.GetListOfKeys()
    #                          if data_type == k.GetName().split('_')[-1]]
    
    self.keys = self.TDir.GetListOfKeys() 
    # self.keys = self.keys[:100]
    
    # if dataset_type != 'data':
    #   self.energies = [0. for k in self.TDir.GetListOfKeys()] # FIXME
    
    if number_keys > 0: self.keys = self.keys[:number_keys]
    
    val_size   = int(np.floor(len(self.keys) * self.val_frac))
    test_size  = int(np.floor(len(self.keys) * self.test_frac))
    train_size = len(self.keys) - val_size - test_size
    
    if self.dataset_type   == 'train':
      self.keys = self.keys[:train_size]
    elif self.dataset_type == 'val':
      self.keys = self.keys[train_size: train_size + val_size]
    elif self.dataset_type == 'test':
      self.keys = self.keys[train_size + val_size:]
      
    # FIXME
    print ('Filtering initial data')
    print (len(self.keys))
    good_indices = [ True for _ in range(len(self.keys)) ]
    for index, key in enumerate(self.keys):
      
      if index % 1000 == 0: print (index, len(self.keys))
      
      tree = key.ReadObj().GetListOfKeys()[-1].ReadObj()
      
      if tree.GetEntriesFast() > 1: good_indices[index] = False
      
      calibfrac = tree.GetMaximum('CalibFrac')
      if calibfrac < 0.5: good_indices[index] = False
      
    self.keys = [k for i, k in enumerate(self.keys) if good_indices[i]]
    print (len(self.keys))
    
    self.on_epoch_end()
    
  def __len__(self):
    return int(np.floor(len(self.keys) / self.batch_size))
  
  def __getitem__(self, batch_index):
    
    low = batch_index * self.batch_size
    high = (batch_index + 1) * self.batch_size
    key_indexes = self.indexes[low:high]
    
    X, Y = self.__data_generation(key_indexes)
    
    return X, Y
  
  def getitembykey(self, batch_index, key):
    
    low = batch_index * self.batch_size
    high = (batch_index + 1) * self.batch_size
    key_indexes = self.indexes[low:high]
    
    X = np.empty((self.batch_size, self.patch_w, self.patch_h, 1))
    
    for sample_num, key_index in enumerate(key_indexes):
      tempDir = self.keys[key_index].ReadObj()
      hist    = tempDir.Get(key)
      if self.n_crop > 0:
        X[sample_num,..., 0] = root_numpy.hist2array(hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
      else:
        X[sample_num,..., 0] = root_numpy.hist2array(hist)
        
      ROOT.SetOwnership(hist, True)
    
    return X
  
  # def getenergy(self, batch_index):
  #   
  #   low = batch_index * self.batch_size
  #   high = (batch_index + 1) * self.batch_size
  #   key_indexes = self.indexes[low:high]
  #   
  #   X = np.empty((self.batch_size, 1))
  #   for sample_num, key_index in enumerate(key_indexes):
  #     X[sample_num, 0] = self.energies[key_index]
  #   return X
    
  def on_epoch_end(self):
    self.indexes = np.arange(len(self.keys))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
    
  def __data_generation(self, key_indexes):
    
    X = np.empty((self.batch_size, self.patch_w, self.patch_h, self.patch_depth))
    Y = np.empty((self.batch_size, self.patch_w, self.patch_h, 1))
    
    for sample_num, key_index in enumerate(key_indexes):
      
      if self.patch_depth == 3:
            
        wire_hist = self.keys[key_index].ReadObj().Get( 'energyCalib' )
        # wire_hist = self.keys[key_index].ReadObj().Get( 'wireCalib' )
        em_hist   = self.keys[key_index].ReadObj().Get( 'cluem' )
        mich_hist = self.keys[key_index].ReadObj().Get( 'clumichel' )
        
        if self.n_crop > 0:
          X[sample_num,..., 0] = root_numpy.hist2array(wire_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
          X[sample_num,..., 1] = root_numpy.hist2array(em_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
          X[sample_num,..., 2] = root_numpy.hist2array(mich_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        else:
          X[sample_num,..., 0] = root_numpy.hist2array(wire_hist)
          X[sample_num,..., 1] = root_numpy.hist2array(em_hist)
          X[sample_num,..., 2] = root_numpy.hist2array(mich_hist)
          
        ROOT.SetOwnership(wire_hist, True)
        ROOT.SetOwnership(em_hist, True)
        ROOT.SetOwnership(mich_hist, True)
      
      elif self.patch_depth == 2:
            
        wire_hist = self.keys[key_index].ReadObj().Get( 'energyCalib' )
        # wire_hist = self.keys[key_index].ReadObj().Get( 'wireCalib' )
        em_hist   = self.keys[key_index].ReadObj().Get( 'cluem' )
        
        if self.n_crop > 0:
          X[sample_num,..., 0] = root_numpy.hist2array(wire_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
          X[sample_num,..., 1] = root_numpy.hist2array(em_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        else:
          X[sample_num,..., 0] = root_numpy.hist2array(wire_hist)
          X[sample_num,..., 1] = root_numpy.hist2array(em_hist)
        
        ROOT.SetOwnership(wire_hist, True)
        ROOT.SetOwnership(em_hist, True)
        
      else:
        wire_hist = self.keys[key_index].ReadObj().Get( 'energyCalib' )
        # wire_hist = self.keys[key_index].ReadObj().Get( 'wireCalib' )
        if self.n_crop > 0:
          X[sample_num, ..., 0] = root_numpy.hist2array(wire_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        else:
          X[sample_num, ..., 0] = root_numpy.hist2array(wire_hist)
        ROOT.SetOwnership(wire_hist, True)
      
      if self.dataset_type != 'data':
        truth_hist = self.keys[key_index].ReadObj().Get( 'truthCalib' )
        if self.n_crop > 0:
          Y[sample_num, ..., 0] = root_numpy.hist2array(truth_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        else:
          Y[sample_num, ..., 0] = root_numpy.hist2array(truth_hist)
        ROOT.SetOwnership(truth_hist, True)
        
    Y[Y > 1.1] = 1.
        
    return X, Y
