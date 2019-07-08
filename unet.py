import tensorflow as tf

import keras
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from keras.preprocessing.image import *
keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')

from losses import *

def unet(inputshape = (160, 160, 3)):

  ##############################################################################
  # Input
  main_input = Input(shape = inputshape, name = 'main_input')
  
  ##############################################################################
  # Downscaling
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(main_input)
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
  pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
  
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
  pool2 = MaxPooling2D(pool_size = ( 2, 2))(conv2)
  
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
  pool3 = MaxPooling2D(pool_size = ( 2, 2))(conv3)
  
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
  pool4 = MaxPooling2D(pool_size = ( 2, 2))(conv4)
  
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
  
  ##############################################################################
  # Upscaling
  up6    = UpSampling2D(size = (2, 2))(conv5)
  up6    = Conv2D(512, 2, activation = 'relu', padding = 'same')(up6)
  merge6 = concatenate([conv4, up6], axis = 3)
  drop6  = Dropout(0.5)(merge6)
  conv6  = Conv2D(512, 3, activation = 'relu', padding = 'same')(drop6)
  conv6  = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
  
  up7    = UpSampling2D(size = (2, 2))(conv6)
  up7    = Conv2D(256, 2, activation = 'relu', padding = 'same')(up7)
  merge7 = concatenate([conv3, up7], axis = 3)
  drop7  = Dropout(0.5)(merge7)
  conv7  = Conv2D(256, 3, activation = 'relu', padding = 'same')(drop7)
  conv7  = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
  
  up8    = UpSampling2D(size = (2, 2))(conv7)
  up8    = Conv2D(128, 2, activation = 'relu', padding = 'same')(up8)
  merge8 = concatenate([conv2, up8], axis = 3)
  drop8  = Dropout(0.5)(merge8)
  conv8  = Conv2D(128, 3, activation = 'relu', padding = 'same')(drop8)
  conv8  = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
  
  up9    = UpSampling2D(size = (2, 2))(conv8)
  up9    = Conv2D(64, 2, activation = 'relu', padding = 'same')(up9)
  merge9 = concatenate([conv1, up9], axis = 3)
  drop9  = Dropout(0.5)(merge9)
  conv9  = Conv2D(64, 3, activation = 'relu', padding = 'same')(drop9)
  conv9  = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
  
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  
  # optimizer = SGD(lr = 0.01, decay = 1E-9, momentum = 0.9, nesterov = True)
  # optimizer = Adam()
  # optimizer = RMSprop(lr = 0.01, rho = 0.9, epsilon = 1e-8, decay = 0.0)
  optimizer = Adadelta(lr = 1.0, rho = 0.95, epsilon = 1e-8, decay = 0.0)
  
  model = Model(inputs = main_input, outputs = conv10)
  model.compile(optimizer = optimizer, loss = loss_jaccard)
  model.summary()

  return model
