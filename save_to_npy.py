# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
import argparse 
parser = argparse.ArgumentParser(description='Run CNN training on patches with' 
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-i', '--input', help = 'Input file')
args = parser.parse_args()

from sys import argv
import os
import numpy as np

import ROOT
from ROOT import TFile
from root_numpy import hist2array

from collections import defaultdict

def main(argv):
    
  meiDir = 'MichelEnergyImage'
  dataTypes = ['wire', 'energy', 'cnnem', 'cnnmichel', 'cluem', 'clumichel', 
               'trueEnergy', 'truth']
    
  for dataType in dataTypes:
        
    data = []
        
    fileName = args.input
    file0 = TFile(fileName)
        
    keys = [k.GetName() for k in file0.Get(meiDir).GetListOfKeys()
            if dataType == k.GetName().split('_')[-1]]
            
    for key in keys: 
      array = hist2array(file0.Get(meiDir + '/' + key))
      data.append(array[2:-2, 2:-2])
    
    if dataType not in os.listdir('.'): os.mkdir(dataType)
    
    output = np.asarray(data)
    np.save(dataType + '/' + dataType, output)
    
    print (dataType, len(output))
    
if __name__ == "__main__":
  main(argv)
