# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
import tensorflow as tf

import keras
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.optimizers import *
from keras.preprocessing.image import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')

import keras_segmentation

from losses import *

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
        
def ConvGroup(n_filt, filt_shape, input_layer, kernel_initializer):
  x = Conv2D(n_filt, filt_shape, padding = 'same',
             kernel_initializer = kernel_initializer,
             bias_initializer = kernel_initializer)(input_layer)
  x = BatchNormalization()(x)
  x = PRELU()(x)
  return x

def unet(inputshape = (160, 160, 1), use_dropout = True, ki = 'he_uniform', 
         conv_depth = 3):
  
  main_input = Input(shape = inputshape, name = 'main_input')
  
  conv1 = ConvGroup(64, 3, main_input, ki)
  
  for i in range(conv_depth - 1): conv1 = ConvGroup(64, 3, conv1, ki) 
    
  pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
  
  conv2 = ConvGroup(128, 3, pool1, ki)
  
  for i in range(conv_depth - 1): conv2 = ConvGroup(128, 3, conv2, ki)
    
  pool2 = MaxPooling2D(pool_size = ( 2, 2))(conv2)
  
  conv3 = ConvGroup(256, 3, pool2, ki)
  
  for i in range(conv_depth - 1): conv3 = ConvGroup(256, 3, conv3, ki)
    
  pool3 = MaxPooling2D(pool_size = ( 2, 2))(conv3)
  
  conv4 = ConvGroup(512, 3, pool3, ki)
  
  for i in range(conv_depth - 1): conv4 = ConvGroup(512, 3, conv4, ki)
    
  if use_dropout:
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size = ( 2, 2))(drop4)
  else:
    pool4 = MaxPooling2D(pool_size = ( 2, 2))(conv4)
  
  conv5 = ConvGroup(1024, 3, pool4, ki)
  
  for i in range(conv_depth - 1): conv5 = ConvGroup(1024, 3, conv5, ki)
  
  if use_dropout:
    drop5 = Dropout(0.5)(conv5)
    up6   = UpSampling2D(size = (2, 2))(drop5)
  else:
    up6 = UpSampling2D(size = (2, 2))(conv5)
  
  up6 = ConvGroup(512, 2, up6, ki)
  
  if use_dropout:
    merge6 = concatenate([drop4, up6], axis = 3)
  else:
    merge6 = concatenate([conv4, up6], axis = 3)
    
  conv6 = ConvGroup(512, 3, merge6, ki)
  
  for i in range(conv_depth - 1): conv6 = ConvGroup(512, 3, conv6, ki)
  
  up7 = UpSampling2D(size = (2, 2))(conv6)
  up7 = ConvGroup(256, 2, up7, ki)
  
  merge7 = concatenate([conv3, up7], axis = 3)
  
  conv7 = ConvGroup(256, 3, merge7, ki)
  
  for i in range(conv_depth - 1): conv7 = ConvGroup(256, 3, conv7, ki)
  
  up8 = UpSampling2D(size = (2, 2))(conv7)
  up8 = ConvGroup(128, 2, up8, ki)
  
  merge8 = concatenate([conv2, up8], axis = 3)
  
  conv8 = ConvGroup(128, 3, merge8, ki)
  
  for i in range(conv_depth - 1): conv8 = ConvGroup(128, 3, conv8, ki)
  
  up9 = UpSampling2D(size = (2, 2))(conv8)
  up9 = ConvGroup(64, 2, up9, ki)
  
  merge9 = concatenate([conv1, up9], axis = 3)
  
  conv9 = ConvGroup(64, 3, merge9, ki)
  
  for i in range(conv_depth - 1): conv9 = ConvGroup(64, 3, conv9, ki)
  
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  
  model = Model(inputs = main_input, outputs = conv10)
  return model

def vgg_unet(inputshape = (160, 160, 1)):
  
  model = keras_segmentation.models.unet.vgg_unet(n_classes = 1, 
                                                  input_height = 160,
                                                  input_width = 160)
  return model
