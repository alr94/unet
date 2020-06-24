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
# keras.backend.set_image_dim_ordering('tf')

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
        
# Default convolutional group types
# Basic 2D conv
def ConvGroup(n_filt, filt_shape, input_layer, kernel_initializer):
  x = Conv2D(n_filt, filt_shape, padding = 'same',
             kernel_initializer = kernel_initializer,
             bias_initializer = kernel_initializer)(input_layer)
  x = BatchNormalization()(x)
  x = PRELU()(x)
  return x

# Inception unit
def InceptionGroup(n_filt, input_layer, kernel_initializer):
  
  stack0 = ConvGroup(n_filt, 1, input_layer, kernel_initializer)
  
  stack1 = ConvGroup(n_filt, 1, input_layer, kernel_initializer)
  stack1 = ConvGroup(n_filt, 3, stack1, kernel_initializer)
  
  stack2 = ConvGroup(n_filt, 1, input_layer, kernel_initializer)
  stack2 = ConvGroup(n_filt, 5, stack2, kernel_initializer)
  
  stack3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_layer)
  stack3 = ConvGroup(n_filt, 1, stack3, kernel_initializer)
  
  out = keras.layers.concatenate([stack0, stack1, stack2, stack3], axis = 3)
  
  return out

def inception_unet(inputshape = (160, 160, 1), ki = 'he_uniform', conv_depth = 3, number_base_nodes = 16):
  
  main_input = Input(shape = inputshape, name = 'main_input')
  
  conv1 = InceptionGroup(number_base_nodes, main_input, ki)
  for i in range(conv_depth - 1): conv1 = InceptionGroup(number_base_nodes, conv1, ki) 
  pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
  
  conv2 = Dropout(0.5)(pool1)
  conv2 = InceptionGroup(2 * number_base_nodes, conv2, ki)
  for i in range(conv_depth - 1): conv2 = InceptionGroup(2 * number_base_nodes, conv2, ki)
  pool2 = MaxPooling2D(pool_size = ( 2, 2))(conv2)
  
  conv3 = Dropout(0.5)(pool2)
  conv3 = InceptionGroup(4 * number_base_nodes, conv3, ki)
  for i in range(conv_depth - 1): conv3 = InceptionGroup(4 * number_base_nodes, conv3, ki)
  pool3 = MaxPooling2D(pool_size = ( 2, 2))(conv3)
  
  conv4 = Dropout(0.5)(pool3)
  conv4 = InceptionGroup(8 * number_base_nodes, conv4, ki)
  for i in range(conv_depth - 1): conv4 = InceptionGroup(8 * number_base_nodes, conv4, ki)
  pool4 = MaxPooling2D(pool_size = ( 2, 2))(conv4)
  
  conv5 = Dropout(0.5)(pool4)
  conv5 = InceptionGroup(16 * number_base_nodes, conv5, ki)
  for i in range(conv_depth - 1): conv5 = InceptionGroup(16 * number_base_nodes,  conv5, ki)
  up6   = UpSampling2D(size = (2, 2))(conv5)
  up6 = InceptionGroup(8 * number_base_nodes, up6, ki)
  merge6 = concatenate([conv4, up6], axis = 3)
    
  conv6 = Dropout(0.5)(merge6)
  conv6 = InceptionGroup(8 * number_base_nodes, conv6, ki)
  for i in range(conv_depth - 1): conv6 = InceptionGroup(8 * number_base_nodes,  conv6, ki)
  up7 = UpSampling2D(size = (2, 2))(conv6)
  up7 = InceptionGroup(4 * number_base_nodes, up7, ki)
  merge7 = concatenate([conv3, up7], axis = 3)
  
  conv7 = Dropout(0.5)(merge7)
  conv7 = InceptionGroup(4 * number_base_nodes, conv7, ki)
  for i in range(conv_depth - 1): conv7 = InceptionGroup(4 * number_base_nodes,  conv7, ki)
  up8 = UpSampling2D(size = (2, 2))(conv7)
  up8 = InceptionGroup(2 * number_base_nodes, up8, ki)
  merge8 = concatenate([conv2, up8], axis = 3)
  
  conv8 = Dropout(0.5)(merge8)
  conv8 = InceptionGroup(2 * number_base_nodes, conv8, ki)
  for i in range(conv_depth - 1): conv8 = InceptionGroup(2 * number_base_nodes,  conv8, ki)
  up9 = UpSampling2D(size = (2, 2))(conv8)
  up9 = InceptionGroup(number_base_nodes, up9, ki)
  merge9 = concatenate([conv1, up9], axis = 3)
  
  conv9 = Dropout(0.5)(merge9)
  conv9 = InceptionGroup(number_base_nodes, conv9, ki)
  for i in range(conv_depth - 1): conv9 = InceptionGroup(number_base_nodes,  conv9, ki)
  
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  
  model = Model(inputs = main_input, outputs = conv10)
  return model

# Previous main model architecture
def unet(inputshape = (160, 160, 1), ki = 'he_uniform', conv_depth = 3):
  
  main_input = Input(shape = inputshape, name = 'main_input')
  
  conv1 = ConvGroup(64, 3, main_input, ki)
  for i in range(conv_depth - 1): conv1 = ConvGroup(64, 3, conv1, ki) 
  conv1 = Dropout(0.5)(conv1)
  pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
  
  conv2 = ConvGroup(128, 3, pool1, ki)
  for i in range(conv_depth - 1): conv2 = ConvGroup(128, 3, conv2, ki)
  conv2 = Dropout(0.5)(conv2)
  pool2 = MaxPooling2D(pool_size = ( 2, 2))(conv2)
  
  conv3 = ConvGroup(256, 3, pool2, ki)
  for i in range(conv_depth - 1): conv3 = ConvGroup(256, 3, conv3, ki)
  conv3 = Dropout(0.5)(conv3)
  pool3 = MaxPooling2D(pool_size = ( 2, 2))(conv3)
  
  conv4 = ConvGroup(512, 3, pool3, ki)
  for i in range(conv_depth - 1): conv4 = ConvGroup(512, 3, conv4, ki)
  conv4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size = ( 2, 2))(conv4)
  
  conv5 = ConvGroup(1024, 3, pool4, ki)
  for i in range(conv_depth - 1): conv5 = ConvGroup(1024, 3, conv5, ki)
  conv5 = Dropout(0.5)(conv5)
  up6   = UpSampling2D(size = (2, 2))(conv5)
  up6 = ConvGroup(512, 2, up6, ki)
  merge6 = concatenate([conv4, up6], axis = 3)
    
  conv6 = ConvGroup(512, 3, merge6, ki)
  for i in range(conv_depth - 1): conv6 = ConvGroup(512, 3, conv6, ki)
  conv6 = Dropout(0.5)(conv6)
  up7 = UpSampling2D(size = (2, 2))(conv6)
  up7 = ConvGroup(256, 2, up7, ki)
  merge7 = concatenate([conv3, up7], axis = 3)
  
  conv7 = ConvGroup(256, 3, merge7, ki)
  for i in range(conv_depth - 1): conv7 = ConvGroup(256, 3, conv7, ki)
  conv7 = Dropout(0.5)(conv7)
  up8 = UpSampling2D(size = (2, 2))(conv7)
  up8 = ConvGroup(128, 2, up8, ki)
  merge8 = concatenate([conv2, up8], axis = 3)
  
  conv8 = ConvGroup(128, 3, merge8, ki)
  for i in range(conv_depth - 1): conv8 = ConvGroup(128, 3, conv8, ki)
  conv8 = Dropout(0.5)(conv8)
  up9 = UpSampling2D(size = (2, 2))(conv8)
  up9 = ConvGroup(64, 2, up9, ki)
  merge9 = concatenate([conv1, up9], axis = 3)
  
  conv9 = ConvGroup(64, 3, merge9, ki)
  for i in range(conv_depth - 1): conv9 = ConvGroup(64, 3, conv9, ki)
  
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  
  model = Model(inputs = main_input, outputs = conv10)
  return model
