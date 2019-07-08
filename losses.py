################################################################################
# setup tensorflow and keras
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

################################################################################
# Model utilities
def intersection(true, pred):
  intersection = K.sum(K.abs(true * pred), axis = [1, 2])
  return intersection

def sum_(true, pred):
  sum_ = K.sum(K.abs(true) + K.abs(pred), axis = [1, 2])
  return sum_

def jaccard(true, pred):
  smooth = 1e-10
  i      = intersection(true, pred)
  s      = sum_(true, pred)
  u      = s - i
  jac    = (i + smooth) / (u + smooth)
  jac    = jac * tf.cast(jac <= 1., jac.dtype)
  return jac

def loss_jaccard(true, pred):
  jac = jaccard(true, pred)
  return - jac
