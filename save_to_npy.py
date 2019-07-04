# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import argparse

from sys import argv
import os
import numpy as np

import ROOT
from ROOT import TFile
from root_numpy import hist2array

from collections import defaultdict

def main(argv):
    
    meiDir = 'MichelEnergyImage'
    dataTypes = ['wire', 'cnn', 'truth']
    
    for dataType in dataTypes:
        
        data = []
        
        for inp in os.listdir('data'):
        
            fileName = 'data/' + inp
            file0 = TFile(fileName)
        
            keys = [k.GetName() for k in file0.Get(meiDir).GetListOfKeys() 
                    if dataType in k.GetName()]
            
            for key in keys: 
                array = hist2array(file0.Get(meiDir + '/' + key))
                data.append(array)
                    
        output = np.asarray(data)
        print (len(output))
        np.save(dataType, output)
    
if __name__ == "__main__":
    main(argv)
