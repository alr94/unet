# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
import argparse 
parser = argparse.ArgumentParser(description = 'Save data from root to npy')
parser.add_argument('-i', '--input',  help = 'Input ROOT file')
parser.add_argument('-o', '--output', help = 'Output Directory')
args   = parser.parse_args()

import os, gc
import multiprocessing as mp
import numpy as np
from collections import defaultdict
import math 

import ROOT
from root_numpy import hist2array

def SaveDataByType(dataType, meiDir, keys, batchSize, batchNumber):
    
  data = []
  for i in range(batchSize):
    
    keyIndex = batchNumber * batchSize + i
    
    if i % 1000 == 0: print (i)
    
    if keyIndex > len(keys) - 1: break
    
    hist = keys[keyIndex].ReadObj()
    data.append(hist2array(hist)[2:-2, 2:-2])
    ROOT.SetOwnership(hist, True)
    
  if data != []:
    dataArray = np.asarray(data)
    np.save(args.output + '/' + dataType + '_' + str(batchNumber), dataArray)
    
if __name__ == "__main__":
  
  batchSize  = 5000
  dataTypes  = ['wire',  'cluem', 'clumichel', 'truth', 'trueEnergy', 'energy', 
                'cnnem', 'cnnmichel',]
  
  file0    = ROOT.TFile(args.input)
  meiDir   = file0.Get('MichelEnergyImage')
  
  for dataType in dataTypes:
    
    print ("Starting data conversion for ", dataType)
  
    keys = [k for k in meiDir.GetListOfKeys()
            if dataType == k.GetName().split('_')[-1]]
    
    n_batches = int(math.ceil(len(keys) / batchSize)) 
    for batchNumber in range(n_batches):
      # Using mp to ensure memory is cleared after each batch
      
      print ("Saving " + dataType + " batch " + str(batchNumber) + '/' + str(n_batches))
      p = mp.Process(target=SaveDataByType, 
                     args=(dataType, meiDir, keys, batchSize, batchNumber,))
      p.start()
      p.join()
