# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import argparse

from sys import argv
import numpy as np

import ROOT
from ROOT import TFile
from root_numpy import hist2array

from collections import defaultdict

def main(argv):
    
    meiDir = 'MichelEnergyImage'
    
    parser = argparse.ArgumentParser(description='Save hists to an array')
    parser.add_argument('-i', '--input', help="Input file", default='input.root')
    args = parser.parse_args()
    
    file0 = TFile(args.input)
    
    dataTypes = ['raw', 'wire', 'cnn', 'truth']
    
    keys = defaultdict(list)
    
    for dataType in dataTypes:
        keys[dataType] = [k.GetName() for k in file0.Get(meiDir).GetListOfKeys() if dataType in k.GetName()]
        
    
    for dataType in dataTypes:
        arrayLen = len(keys[dataType])
        for key in keys[dataType]: 
            array = hist2array(file0.Get(meiDir + '/' + key))
        
    
if __name__ == "__main__":
    main(argv)
