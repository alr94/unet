# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
import tensorflow as tf

import keras_segmentation

import keras
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from keras.preprocessing.image import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')

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

def unet(inputshape = (160, 160, 1)):

  use_dropout        = True 
  kernel_initializer = 'he_uniform'
  conv_depth = 2
  
  main_input = Input(shape = inputshape, name = 'main_input')
  
  conv1 = ConvGroup(64, 3, main_input, kernel_initializer)
  # conv1 = Conv2D(64, 3, padding = 'same',
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(main_input)
  # conv1 = BatchNormalization()(conv1)
  # conv1 = PRELU()(conv1)
  
  for i in range(conv_depth - 1):
    conv1 = ConvGroup(64, 3, conv1, kernel_initializer) 
    # conv1 = Conv2D(64, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv1)
    # conv1 = BatchNormalization()(conv1)
    # conv1 = PRELU()(conv1)
    
  pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
  
  conv2 = ConvGroup(128, 3, pool1, kernel_initializer)
  # conv2 = Conv2D(128, 3, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(pool1)
  # conv2 = BatchNormalization()(conv2)
  # conv2 = PRELU()(conv2)
  
  for i in range(conv_depth - 1):
    conv2 = ConvGroup(128, 3, conv2, kernel_initializer)
    # conv2 = Conv2D(128, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv2)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = PRELU()(conv2)
    
  pool2 = MaxPooling2D(pool_size = ( 2, 2))(conv2)
  
  conv3 = ConvGroup(256, 3, pool2, kernel_initializer)
  # conv3 = Conv2D(256, 3, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(pool2)
  # conv3 = BatchNormalization()(conv3)
  # conv3 = PRELU()(conv3)
  
  for i in range(conv_depth - 1):
    conv3 = ConvGroup(256, 3, conv3, kernel_initializer)
    # conv3 = Conv2D(256, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv3)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = PRELU()(conv3)
    
  pool3 = MaxPooling2D(pool_size = ( 2, 2))(conv3)
  
  conv4 = ConvGroup(512, 3, pool3, kernel_initializer)
  # conv4 = Conv2D(512, 3, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(pool3)
  # conv4 = BatchNormalization()(conv4)
  # conv4 = PRELU()(conv4)
  
  for i in range(conv_depth - 1):
    conv4 = ConvGroup(512, 3, conv4, kernel_initializer)
    # conv4 = Conv2D(512, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv4)
    # conv4 = BatchNormalization()(conv4)
    # conv4 = PRELU()(conv4)
    
  if use_dropout:
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size = ( 2, 2))(drop4)
  else:
    pool4 = MaxPooling2D(pool_size = ( 2, 2))(conv4)
  
  conv5 = ConvGroup(1024, 3, pool4, kernel_initializer)
  # conv5 = Conv2D(1024, 3, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(pool4)
  # conv5 = BatchNormalization()(conv5)
  # conv5 = PRELU()(conv5)
  
  for i in range(conv_depth - 1):
    conv5 = ConvGroup(1024, 3, conv5, kernel_initializer)
    # conv5 = Conv2D(1024, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv5)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = PRELU()(conv5)
  
  if use_dropout:
    drop5 = Dropout(0.5)(conv5)
    up6    = UpSampling2D(size = (2, 2))(drop5)
  else:
    up6    = UpSampling2D(size = (2, 2))(conv5)
  
  up6 = ConvGroup(512, 2, up6, kernel_initializer)
  # up6    = Conv2D(512, 2, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(up6)
  # up6 = BatchNormalization()(up6)
  # up6 = PRELU()(up6)
  
  if use_dropout:
    merge6 = concatenate([drop4, up6], axis = 3)
  else:
    merge6 = concatenate([conv4, up6], axis = 3)
    
  conv6 = ConvGroup(512, 3, merge6, kernel_initializer)
  # conv6  = Conv2D(512, 3, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(merge6)
  # conv6 = BatchNormalization()(conv6)
  # conv6 = PRELU()(conv6)
  
  for i in range(conv_depth - 1):
    conv6 = ConvGroup(512, 3, conv6, kernel_initializer)
    # conv6  = Conv2D(512, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv6)
    # conv6 = BatchNormalization()(conv6)
    # conv6 = PRELU()(conv6)
  
  up7    = UpSampling2D(size = (2, 2))(conv6)
  up7 = ConvGroup(256, 2, up7, kernel_initializer)
  # up7    = Conv2D(256, 2, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(up7)
  # up7 = BatchNormalization()(up7)
  # up7 = PRELU()(up7)
  
  merge7 = concatenate([conv3, up7], axis = 3)
  
  conv7 = ConvGroup(256, 3, merge7, kernel_initializer)
  # conv7  = Conv2D(256, 3, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(merge7)
  # conv7 = BatchNormalization()(conv7)
  # conv7 = PRELU()(conv7)
  
  for i in range(conv_depth - 1):
    conv7 = ConvGroup(256, 3, conv7, kernel_initializer)
    # conv7  = Conv2D(256, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv7)
    # conv7 = BatchNormalization()(conv7)
    # conv7 = PRELU()(conv7)
  
  up8    = UpSampling2D(size = (2, 2))(conv7)
  up8 = ConvGroup(128, 2, up8, kernel_initializer)
  # up8    = Conv2D(128, 2, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(up8)
  # up8 = BatchNormalization()(up8)
  # up8 = PRELU()(up8)
  
  merge8 = concatenate([conv2, up8], axis = 3)
  
  conv8 = ConvGroup(128, 3, merge8, kernel_initializer)
  # conv8  = Conv2D(128, 3, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(merge8)
  # conv8 = BatchNormalization()(conv8)
  # conv8 = PRELU()(conv8)
  
  for i in range(conv_depth - 1):
    conv8 = ConvGroup(128, 3, conv8, kernel_initializer)
    # conv8  = Conv2D(128, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv8)
    # conv8 = BatchNormalization()(conv8)
    # conv8 = PRELU()(conv8)
  
  up9    = UpSampling2D(size = (2, 2))(conv8)
  up9 = ConvGroup(64, 2, up9, kernel_initializer)
  # up9    = Conv2D(64, 2, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(up9)
  # up9 = BatchNormalization()(up9)
  # up9 = PRELU()(up9)
  
  merge9 = concatenate([conv1, up9], axis = 3)
  
  conv9 = ConvGroup(64, 3, merge9, kernel_initializer)
  # conv9  = Conv2D(64, 3, padding = 'same', 
  #                kernel_initializer = kernel_initializer,
  #                bias_initializer = kernel_initializer)(merge9)
  # conv9 = BatchNormalization()(conv9)
  # conv9 = PRELU()(conv9)
  
  for i in range(conv_depth - 1):
    conv9 = ConvGroup(64, 3, conv9, kernel_initializer)
    # conv9  = Conv2D(64, 3, padding = 'same', 
    #                kernel_initializer = kernel_initializer,
    #                bias_initializer = kernel_initializer)(conv9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = PRELU()(conv9)
  
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  
  model = Model(inputs = main_input, outputs = conv10)
  return model

def vgg_unet(inputshape = (160, 160, 1)):
  
  model = keras_segmentation.models.unet.vgg_unet(n_classes = 1, 
                                                  input_height = 160,
                                                  input_width = 160)
  return model
