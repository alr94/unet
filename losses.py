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
# keras.backend.set_image_dim_ordering('tf')

################################################################################
# Model utilities
def intersection(true, pred):
  intersection = K.sum(K.abs(true * pred), axis = [1, 2])
  return intersection

def sum_(true, pred):
  sum_ = K.sum(K.abs(true) + K.abs(pred), axis = [1, 2])
  return sum_

def union(true, pred):
  sum_ = K.sum(K.abs(true) + K.abs(pred), axis = [1, 2])
  intersection = K.sum(K.abs(true * pred), axis = [1, 2])
  return sum_ - intersection

def jaccard(true, pred, smooth):
  i      = intersection(true, pred)
  u      = union(true, pred)
  jac    = (i + smooth) / (u + smooth)
  jac    = jac * tf.cast(jac <= 1., jac.dtype)
  return jac

def loss_jaccard(true, pred):
  smooth = 1e-10
  jac = jaccard(true, pred, smooth)
  return - jac

def jaccard_distance(y_true, y_pred, smooth=1e-10):
  intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
  sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
  jac = (intersection + smooth) / (sum_ - intersection + smooth)
  return tf.reduce_mean(- jac)
