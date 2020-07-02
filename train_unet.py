# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-i' , '--input' , help = 'Input list')
parser.add_argument('-w' , '--weights'   , help = 'Weights file (optional)')
parser.add_argument('-m' , '--model'     , help = 'Model file (optional)')
parser.add_argument('-g' , '--gpu'       , help = 'Which GPU index'          , default = '0')
parser.add_argument('-e' , '--epochs'    , help = 'Number of epochs'         , type = int                  ,
                    default = 10)
parser.add_argument('-l', '--loss',    help = 'Desired loss', type = float)
parser.add_argument('-s', '--save',    help = 'Save Model?')

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
# keras.backend.set_image_dim_ordering('tf')
        
################################################################################
# Other setups
import numpy as np
import datetime
from random import shuffle

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

################################################################################
# My stuff
from utils import get_unet_data
from losses import *
from unet import *
from data_gen import DataGenerator

################################################################################

def SaveModel(model, name):
    name += '_'
    name += datetime.datetime.now().strftime("%y%m%d-%H:%M")
    with open(name + '_architecture.json', 'w') as f: 
      f.write(model.to_json())
    model.save_weights(name + '_weights.h5', overwrite = True)
    
def FilterData(x, y):
  r = loss_jaccard(K.variable(y), 
                   K.variable(y)).eval(session = K.get_session())
  
  idx = np.any(r < - 1e-10, axis = 1)
  x = x[idx]
  y = y[idx]
  
  return x, y
  
def TrainOnBatch(batch, args, model, val_x, val_y, batch_size, n_channels):
  
  print ('Getting traing data batch ', str(batch))
  
  train_x, train_y = get_unet_data(args.input, batch, n_channels)
  
  n_patches, patch_w, patch_h, patch_depth = train_x.shape

  print ('Fitting training data batch ', str(batch))
  h = model.fit(train_x, train_y, validation_data = (val_x, val_y),
                batch_size = batch_size,
                shuffle = True, epochs = 1)
  
  return h

def TrainWithGenerator(train_gen, tb_callback, model, epochs = 1, 
                       val_gen = None):
  print ('Fitting with generator')
  if not val_gen:
    h = model.fit_generator(train_gen, shuffle = True, epochs = epochs,
                            callbacks = [ tb_callback ])
  else:
    h = model.fit_generator(train_gen, validation_data = val_gen, 
                            shuffle = True, epochs = epochs,
                            callbacks = [ tb_callback ])
  return h

################################################################################
use_generator     = True
model_type = 'inception'

batch_size  = 1
test_sample = False

n_channels        = 1
conv_depth        = 4
number_base_nodes = 24
number_layers     = 5

patch_w, patch_h, patch_depth = 160, 160, n_channels 

name = model_type 
name += '_basenodes' + str(number_base_nodes)
name += '_layers' + str(number_layers)
name += '_convdepth' + str(conv_depth)
name += '_patchsize' + str(patch_w)
name += '_final'
################################################################################

if use_generator:
  print ('Using generator')
  
  number_keys = 100 if test_sample else 0
  
  train_gen = DataGenerator(root_data = args.input, 
                            dataset_type = 'train', 
                            dirname = 'MichelEnergyImage', 
                            batch_size = batch_size, 
                            patch_w = patch_w, 
                            patch_h = patch_h, 
                            patch_depth = n_channels,
                            number_keys = number_keys)
    
  val_gen   = DataGenerator(root_data = args.input, 
                            dataset_type = 'val', 
                            dirname = 'MichelEnergyImage', 
                            batch_size = batch_size, 
                            patch_w = patch_w, 
                            patch_h = patch_h, 
                            patch_depth = n_channels,
                            number_keys = number_keys)
  
else:
  print ('Using large batches')
  available_batches = [int(val.split('.')[0].split('_')[1]) for val in 
                       os.listdir(args.input) if 'wire' in val]
  
  test_batches  = available_batches[-1]
  val_batches   = available_batches[-2]
  train_batches = available_batches[:-2]
  
  val_x,  val_y  = get_unet_data(args.input, val_batches, n_channels)
  n_val, patch_w, patch_h, patch_depth = val_x.shape

################################################################################
# Training
sess = tf.InteractiveSession()
with sess.as_default():
  
  # Check loss
  # for i in range(100):
  #   X, Y = train_gen.__getitem__(i)
  #   zeros = np.zeros(Y.shape)
  #   loss = jaccard_distance(Y, zeros)
  #   print (loss.eval())
  # exit()
  
  print ('Building model')
  if args.model: 
    model_json      = open(args.model, 'r')
    read_model_json = model_json.read()
    model_json.close()
    model = model_from_json(read_model_json)
  else: 
    if model_type == 'vgg': 
      model = vgg_unet(inputshape = (patch_w, patch_h, patch_depth))
    elif model_type == 'inception': 
      model = inception_unet(inputshape = (patch_w, patch_h, patch_depth), 
                             conv_depth=conv_depth, 
                             number_base_nodes = number_base_nodes, 
                             number_layers = number_layers)
    else: 
      model = unet(inputshape = (patch_w, patch_h, patch_depth))
  
  optimizer = Nadam()
  
  model.compile(optimizer = optimizer, loss = jaccard_distance,
                metrics = [efficiency, purity])
  model.summary()

  if args.weights: 
    print ('Loading initial weights')
    model.load_weights(args.weights)
  
  loss, losses         = 0., []
  val_loss, val_losses = 0., []
  epoch                = 0
  
  if use_generator:
    
    if args.loss:
      
      tb_callback = keras.callbacks.TensorBoard(log_dir='log/' + name)
      
      while epoch < args.epochs and loss > float(args.loss):
        
        epoch += 1
        print ("Epoch: ", epoch, "of ", args.epochs)
  
        # if epoch > 1: optimizer.lr.assign(0.01)
        
          
        h = TrainWithGenerator(train_gen, tb_callback, model, epochs = 1, 
                               val_gen = val_gen)
        
        loss = h.history['loss'][0]
        losses.append(loss)
        
        val_loss = h.history['val_loss'][0]
        val_losses.append(val_loss)
        
        if args.save == 'y':
          print ('Saving model checkpoint')
          SaveModel(model, 
                    name + '_epoch' + str(epoch) + 
                    '_loss' + str(loss) + 
                    '_val' + str(val_loss)
                   ) 
          
    else:
      
      tb_callback = keras.callbacks.TensorBoard(log_dir='log/' + name)
      
      while epoch < args.epochs:
        
        epoch += 1
        print ("Epoch: ", epoch, "of ", args.epochs)
        
        # if epoch > 1: optimizer.lr.assign(0.01)
          
        h = TrainWithGenerator(train_gen, tb_callback, model, epochs = 1, 
                               val_gen = val_gen)
        
        loss = h.history['loss'][0]
        losses.append(loss)
        
        val_loss = h.history['val_loss'][0]
        val_losses.append(val_loss)
        
        if args.save == 'y':
          print ('Saving model checkpoint')
          SaveModel(model, 
              name + '_epoch' + str(epoch) + 
                    '_loss' + str(loss) + 
                    '_val' + str(val_loss)
                   ) 
  else:
    if args.loss:
      
      while epoch < args.epochs and loss > float(args.loss):
        
        epoch += 1
        print ("Epoch: ", epoch, "of ", args.epochs)
        
        # if epoch > 1: optimizer.lr.assign(0.01)
        
        shuffle(train_batches)
        for batch in train_batches:
          h = TrainOnBatch(batch, args, model, val_x, val_y, batch_size, 
                           n_channels)
          
          loss = h.history['loss'][0]
          losses.append(loss)
          
          val_loss = h.history['val_loss'][0]
          val_losses.append(val_loss)
          
        if args.save == 'y':
          print ('Saving model checkpoint')
          SaveModel(model, 
              name + '_epoch' + str(epoch) + 
                    '_loss' + str(loss) + 
                    '_val' + str(val_loss)
                   ) 
    else:
      while epoch < args.epochs:
        
        epoch += 1
        print ("Epoch: ", epoch, "of ", args.epochs)
        
        # if epoch > 1: optimizer.lr.assign(0.01)
        
        shuffle(train_batches)
        for batch in train_batches:
          h = TrainOnBatch(batch, args, model, val_x, val_y, batch_size, 
                           n_channels)
          
          loss = h.history['loss'][0]
          losses.append(loss)
          
          val_loss = h.history['val_loss'][0]
          val_losses.append(val_loss)
        
        if args.save == 'y':
          print ('Saving model checkpoint')
          SaveModel(model, 
              name + '_epoch' + str(epoch) + 
                    '_loss' + str(loss) + 
                    '_val' + str(val_loss)
                   ) 
        
if args.save == 'y':
  print ('Saving model checkpoint')
  SaveModel(model, 
      name + '_epoch' + str(epoch) + 
            '_loss' + str(loss) + 
            '_val' + str(val_loss)
           ) 

if use_generator:
  
  print ('Evaluating model on test set')
  test_gen = DataGenerator(root_data = args.input, 
                           dataset_type = 'test', 
                           dirname = 'MichelEnergyImage', 
                           batch_size = batch_size, 
                           patch_w = patch_w, 
                           patch_h = patch_h, 
                           patch_depth = n_channels)
  score = model.evaluate_generator(test_gen)
  print (score)

  # test_x = np.zeros((batch_size * test_gen.__len__(), patch_w, patch_h, 
  #   n_channels))
  # test_y = np.zeros((batch_size * test_gen.__len__(), patch_w, patch_h, 1))
  # 
  # for i in range(test_gen.__len__()):
  #   x, y = test_gen.__getitem__(i)
  #   test_x[i * batch_size: (i + 1) * batch_size] = x
  #   test_y[i * batch_size: (i + 1) * batch_size] = y
    
  plot = plt.plot(losses)
  plt.savefig('img/losses.png')
  plt.close()
  
  plot = plt.plot(val_losses)
  plt.savefig('img/val_losses.png')
  plt.close()
  
else:
  print ('Evaluating model on test set')
  test_x, test_y = get_unet_data(args.input, test_batches, n_channels)
  score = model.evaluate(test_x, test_y)
  print ('Test score: ', score)
    
  plot = plt.plot(losses)
  plt.savefig('img/losses.png')
  plt.close()
  
  plot = plt.plot(val_losses)
  plt.savefig('img/val_losses.png')
  plt.close()
