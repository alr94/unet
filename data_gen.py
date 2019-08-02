# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
import numpy as np
import keras

import ROOT
import root_numpy

from collections import defaultdict
import itertools

from losses import *

class DataGenerator(keras.utils.Sequence):

  def __init__(self, root_data, dataset_type, batch_size = 32, shuffle = True, 
               patch_w = 160, patch_h = 160, patch_depth = 3, val_frac = 0.05,
               test_frac = 0.05):
    
    self.dataset_type = dataset_type
    self.val_frac  = val_frac
    self.test_frac = test_frac
    
    self.batch_size   = batch_size
    self.shuffle      = shuffle
    
    self.dim         = (patch_w, patch_h, patch_depth)
    self.patch_w     = patch_w
    self.patch_h     = patch_h
    self.patch_depth = patch_depth
    
    self.n_crop = int((164 - patch_w)/ 2)
    
    self.TFile = ROOT.TFile(root_data)
    self.TDir  = self.TFile.Get('MichelEnergyImage')
    
    self.data_types = ['wire', 'cluem', 'clumichel', 'truth']
    
    self.keys = defaultdict()
    for data_type in self.data_types:
      self.keys[data_type] = [k for k in self.TDir.GetListOfKeys()
                              if data_type == k.GetName().split('_')[-1]]
    
    val_size   = int(np.floor(len(self.keys['truth']) * self.val_frac))
    test_size  = int(np.floor(len(self.keys['truth']) * self.test_frac))
    train_size = len(self.keys['truth']) - val_size - test_size
    
    if self.dataset_type   == 'train':
      for data_type in self.data_types:
        self.keys[data_type] = self.keys[data_type][:train_size]
    elif self.dataset_type == 'val':
      for data_type in self.data_types:
        self.keys[data_type] = self.keys[data_type][train_size:
                                                    train_size + val_size]
    elif self.dataset_type == 'test':
      for data_type in self.data_types:
        self.keys[data_type] = self.keys[data_type][train_size + val_size:]
      
    print ('Filtering initial data')
    nan_index = [ True for i in range(len(self.keys['wire'])) ]
    for data_type in self.data_types:
      print(data_type)
      for i in range(len(self.keys[data_type])):
        key        = self.keys[data_type][i]
        test_hist  = key.ReadObj()
        test_array = root_numpy.hist2array(test_hist)[self.n_crop:-self.n_crop, 
                                                      self.n_crop:-self.n_crop]
        ROOT.SetOwnership(test_hist, True)
        if np.isnan(np.sum(test_array)): nan_index[i] = False
          
    for data_type in self.data_types:
      self.keys[data_type] =  [d for d, s in 
                               itertools.izip(self.keys[data_type], nan_index) 
                               if s]
    
    self.on_epoch_end()
    
  def __len__(self):
    return int(np.floor(len(self.keys['truth']) / self.batch_size))
  
  def __getitem__(self, batch_index):
    
    low = batch_index * self.batch_size
    high = (batch_index + 1) * self.batch_size
    key_indexes = self.indexes[low:high]
    
    X, Y = self.__data_generation(key_indexes)
    
    return X, Y
    
  def on_epoch_end(self):
    self.indexes = np.arange(len(self.keys['truth']))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
    
  def __data_generation(self, key_indexes):
    
    X = np.empty((self.batch_size, self.patch_w, self.patch_h, self.patch_depth))
    Y = np.empty((self.batch_size, self.patch_w, self.patch_h, 1))
    
    for sample_num, key_index in enumerate(key_indexes):
      
      if self.patch_depth == 3:
            
        wire_hist = self.keys['wire'][key_index].ReadObj()
        em_hist   = self.keys['cluem'][key_index].ReadObj()
        mich_hist = self.keys['clumichel'][key_index].ReadObj()
        
        X[sample_num,..., 0] = root_numpy.hist2array(wire_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        X[sample_num,..., 1] = root_numpy.hist2array(em_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        X[sample_num,..., 2] = root_numpy.hist2array(mich_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        
        ROOT.SetOwnership(wire_hist, True)
        ROOT.SetOwnership(em_hist, True)
        ROOT.SetOwnership(mich_hist, True)
      
      elif self.patch_depth == 2:
            
        wire_hist = self.keys['wire'][key_index].ReadObj()
        em_hist   = self.keys['cluem'][key_index].ReadObj()
        
        X[sample_num,..., 0] = root_numpy.hist2array(wire_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        X[sample_num,..., 1] = root_numpy.hist2array(em_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        
        ROOT.SetOwnership(wire_hist, True)
        ROOT.SetOwnership(em_hist, True)
        
      else:
        wire_hist = self.keys['wire'][key_index].ReadObj()
        X[sample_num, ..., 0] = root_numpy.hist2array(wire_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
        ROOT.SetOwnership(wire_hist, True)
      
      truth_hist = self.keys['truth'][key_index].ReadObj()
      Y[sample_num, ..., 0] = root_numpy.hist2array(truth_hist)[self.n_crop:-self.n_crop , self.n_crop:-self.n_crop]
      ROOT.SetOwnership(truth_hist, True)
      
    return X, Y
