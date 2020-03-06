# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-i', '--input',   help = 'Input directory')
parser.add_argument('-w', '--weights', help = 'Weights file (optional)')
parser.add_argument('-m', '--model',   help = 'Model file (optional)')
parser.add_argument('-g', '--gpu',     help = 'Which GPU index', default = '0')
parser.add_argument('-e', '--epochs',  help = 'Number of epochs', type = int, 
                    default = 5)
parser.add_argument('-l', '--loss',    help = 'Desired loss', type = float)
parser.add_argument('-n', '--name',    help = 'Human Name')

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
# Build dataset generators
use_vgg       = False
use_generator = args.input.split('.')[-1] == 'root'
batch_size    = 8
n_channels    = 3

patch_w, patch_h, patch_depth = 160, 160, n_channels 

################################################################################ 

if use_generator:
  print ('Using generator')
  
  train_gen = DataGenerator(root_data = args.input, 
                            dataset_type = 'train', 
                            dirname = 'MichelEnergyImage', 
                            batch_size = batch_size, 
                            patch_w = patch_w, 
                            patch_h = patch_h, 
                            patch_depth = n_channels)
  
  val_gen   = DataGenerator(root_data = args.input, 
                            dataset_type = 'val', 
                            dirname = 'MichelEnergyImage', 
                            batch_size = batch_size, 
                            patch_w = patch_w, 
                            patch_h = patch_h, 
                            patch_depth = n_channels)
  
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
  
  print ('Building model')
  if args.model: 
    model_json      = open(args.model, 'r')
    read_model_json = model_json.read()
    model_json.close()
    model = model_from_json(read_model_json)
  else: 
    if use_vgg:
      model = vgg_unet(inputshape = (patch_w, patch_h, patch_depth))
    else:
      model = unet(inputshape = (patch_w, patch_h, patch_depth))
  
  # optimizer = SGD(lr = 0.01)
  # optimizer = SGD(lr = 0.1, decay = 1E-9, momentum = 0.9, nesterov = True)
  # optimizer = Adam()
  optimizer = Nadam()
  # optimizer = RMSprop(lr = 0.01, rho = 0.9, epsilon = 1e-8, decay = 0.0)
  # optimizer = Adadelta(lr = 1.0, rho = 0.95, epsilon = 1e-8, decay = 0.0)
  
  model.compile(optimizer = optimizer, loss = jaccard_distance)
  model.summary()

  if args.weights: 
    print ('Loading initial weights')
    model.load_weights(args.weights)
  
  loss, losses         = 0., []
  val_loss, val_losses = 0., []
  epoch                = 0
  
  if use_generator:
    
    if args.loss:
      
      tb_callback = keras.callbacks.TensorBoard(log_dir='log/' + args.name)
      
      while epoch < args.epochs and loss > args.loss:
        
        epoch += 1
        print ("Epoch: ", epoch, "of ", args.epochs)
        
        h = TrainWithGenerator(train_gen, tb_callback, model, epochs = 1, 
                               val_gen = val_gen)
        
        loss = h.history['loss'][0]
        losses.append(loss)
        
        val_loss = h.history['val_loss'][0]
        val_losses.append(val_loss)
        
        print ('Saving model checkpoint')
        SaveModel(model, 'model_epoch' + str(epoch) + '_loss' + str(loss) + 
                  '_val' + str(val_loss)) 
          
    else:
      
      tb_callback = keras.callbacks.TensorBoard(log_dir='log/' + args.name)
      
      while epoch < args.epochs:
        
        epoch += 1
        print ("Epoch: ", epoch, "of ", args.epochs)
        
        h = TrainWithGenerator(train_gen, tb_callback, model, epochs = 1, 
                               val_gen = val_gen)
        
        loss = h.history['loss'][0]
        losses.append(loss)
        
        val_loss = h.history['val_loss'][0]
        val_losses.append(val_loss)
        
        print ('Saving model checkpoint')
        SaveModel(model, 'model_epoch' + str(epoch) + '_loss' + str(loss) + 
                  '_val' + str(val_loss)) 
      
  else:
    if args.loss:
      
      while epoch < args.epochs and loss > args.loss:
        
        epoch += 1
        print ("Epoch: ", epoch, "of ", args.epochs)
        
        shuffle(train_batches)
        for batch in train_batches:
          h = TrainOnBatch(batch, args, model, val_x, val_y, batch_size, 
                           n_channels)
          
          loss = h.history['loss'][0]
          losses.append(loss)
          
          val_loss = h.history['val_loss'][0]
          val_losses.append(val_loss)
          
        print ('Saving model checkpoint')
        SaveModel(model, 'model_epoch' + str(epoch) + '_loss' + str(loss) + 
                  '_val' + str(val_loss)) 
          
    else:
      while epoch < args.epochs:
        
        epoch += 1
        print ("Epoch: ", epoch, "of ", args.epochs)
        
        shuffle(train_batches)
        for batch in train_batches:
          h = TrainOnBatch(batch, args, model, val_x, val_y, batch_size, 
                           n_channels)
          
          loss = h.history['loss'][0]
          losses.append(loss)
          
          val_loss = h.history['val_loss'][0]
          val_losses.append(val_loss)
        
        print ('Saving model checkpoint')
        SaveModel(model, 'model_epoch' + str(epoch) + '_loss' + str(loss) + 
                  '_val' + str(val_loss)) 
        
print ('Saving model checkpoint')
SaveModel(model, 'model_epoch' + str(epoch) + '_loss' + str(loss) + 
          '_val' + str(val_loss)) 

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

  test_x = np.zeros((batch_size * test_gen.__len__(), patch_w, patch_h, 
    n_channels))
  test_y = np.zeros((batch_size * test_gen.__len__(), patch_w, patch_h, 1))
  
  for i in range(test_gen.__len__()):
    x, y = test_gen.__getitem__(i)
    test_x[i * batch_size: (i + 1) * batch_size] = x
    test_y[i * batch_size: (i + 1) * batch_size] = y
    
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
