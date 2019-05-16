# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2

################################################################################
# Parsing of input file and arguments
import argparse 

parser = argparse.ArgumentParser(description='Run CNN training on patches with' 
                                 + ' a few different hyperparameter sets.')

parser.add_argument('-c', '--config', help = 'JSON with script configuration')
parser.add_argument('-o', '--output', help = 'Output model file name')
parser.add_argument('-g', '--gpu',    help = 'Which GPU index', default = '0')

args = parser.parse_args()

################################################################################
# setup tensorflow enviroment variables
import os
from os.path import exists, isfile, join

os.environ['KERAS_BACKEND']        = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

################################################################################
# setup tensorflow and keras
import tensorflow as tf
print 'Using Tensorflow version: ', tf.__version__

import keras
from keras.models import Model
from keras.layers import Input, concatenate, concatenate
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD

print 'Using Keras version: ', keras.__version__

keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')

# These classes solve issues with the save_model function for PReLU/LeakyReLU 
# https://github.com/keras-team/keras/issues/3816
class PRELU(PReLU):
  def __init__(self, **kwargs):
    self.__name__ = "PReLU"
    super(PRELU, self).__init__(**kwargs)
                            
class LEAKYRELU(LeakyReLU):
  def __init__(self, **kwargs):
    self.__name__ = "LeakyReLU"
    super(LEAKYRELU, self).__init__(**kwargs)
        
################################################################################
# Other setups
import numpy as np
import json
from utils import read_config, get_patch_size, count_events
import math
import datetime

################################################################################
# Additional utils
def save_model(model, name):
try:
    name += '_'
    name += datetime.datetime.now().strftime("%y%m%d-%H:%M")
    with open(name + '_architecture.json', 'w') as f: f.write(model.to_json())
    model.save_weights(name + '_weights.h5', overwrite=True)
return True
except: return False

################################################################################
# Get data

for directory in os.listdir(INPUT_DIR):

    print 'Reading data from', directory
    
    fnameX = [f for f in os.listdir(INPUT_DIR + '/' + directory) 
              if '_x.npy' in f]
    fnameY = [f for f in os.listdir(INPUT_DIR + '/' + directory) 
              if '_y.npy' in f]
    
    if len(fnameX) != 1 or len(fnameY) != 1: continue 

