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
  
  # FIXME: test
  # y_pred[y_pred > 0.5] = 1.
  # y_pred[y_pred < 0.5] = 0.
  
  intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
  sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
  
  # i_eval = intersection.eval()
  # s_eval = intersection.eval()
  # print (i_eval)
  # print (s_eval)
  
  # for i in range(len(i_eval)):
  #   if i_eval[i] > s_eval[i] / 2:
  #     for x in range(len(y_true[i])):
  #       for y in range(len(y_true[i][x])):
  #         if y_true[i][x,y] > 1.1: 
  #           print (y_true[i][x,y])#, end = ' ')
      
  jac = (intersection + smooth) / (sum_ - intersection + smooth)
  return tf.reduce_mean(- jac)

def efficiency(y_true, y_pred):
  
  # y_pred[y_pred > 0.5] = 1.
  # y_pred[y_pred < 0.5] = 0.
  
  smooth = 1e-10
  intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
  sum_ = tf.reduce_sum(y_true, axis=(1,2))
  eff = (intersection + smooth) / (sum_ + smooth)
  return tf.reduce_mean(eff)

def purity(y_true, y_pred):
  
  # y_pred[y_pred > 0.5] = 1.
  # y_pred[y_pred < 0.5] = 0.
  
  smooth = 1e-10
  intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
  sum_ = tf.reduce_sum(y_pred, axis=(1,2))
  purity = (intersection + smooth) / (sum_ + smooth)
  return tf.reduce_mean(purity)

def pur_eff(y_true, y_pred):
  pur = purity(y_true, y_pred)
  eff = efficiency(y_true, y_pred)
  return - pur * eff

def f_beta(y_true, y_pred):
  beta = 2
  pur = purity(y_true, y_pred)
  eff = efficiency(y_true, y_pred)
  f = (1 + beta * beta) * (pur * eff) / (beta * beta * pur + eff)
  return -f
