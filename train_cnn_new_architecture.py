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
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True
    except: return False

################################################################################
# Configurations
print 'Reading configurations...'

config = read_config(args.config)

INPUT_DIR = config['training_on_patches']['input_dir']

PATCH_SIZE_W, PATCH_SIZE_D = get_patch_size(INPUT_DIR)

# General 
FRACTION_VALIDATION = 0.01
FRACTION_TEST       = 0.01
BATCH_SIZE          = config['training_on_patches']['batch_size']
N_CLASSES           = config['training_on_patches']['nb_classes']
N_EPOCH             = config['training_on_patches']['nb_epoch']

# Network 
N_FILTERS1   = 16
N_CONV1      = 1
N_FILTERS2   = 16
N_CONV2      = 3
N_FILTERS3   = 16
N_CONV3      = 5
CONV_ACT_FN1 = 'relu'
DROPOUT1     = 0.2

DENSESIZE1   = 128
DENS_ACT_FN1 = 'relu'

DENSESIZE2   = 32
DENS_ACT_FN2 = 'relu'

DROPOUT2     = 0.2

CONFIG_NAME  = 'inception_' + CONV_ACT_FN1 + '_' + str(N_EPOCH) + '_epoch'

datestring = datetime.datetime.now().strftime("%y%m%d-%h:%m")
log_dir = './tb/' + CONFIG_NAME + '--' + datestring

################################################################################
# CNN Definition
print 'Defining CNN model...'

with tf.device('/gpu:' + args.gpu):
    
    with tf.name_scope('input'): 
        main_input = Input(shape=(PATCH_SIZE_W, PATCH_SIZE_D, 1), 
                           name='main_input')
    with tf.name_scope('conv1'):
        x1 = Conv2D(N_FILTERS1, (N_CONV1, N_CONV1), data_format='channels_last',
                activation='relu', border_mode='same')(main_input)
        x3 = Conv2D(N_FILTERS2, (N_CONV2, N_CONV2), data_format='channels_last',
                activation='relu', border_mode='same')(main_input)
        x5 = Conv2D(N_FILTERS3, (N_CONV3, N_CONV3), data_format='channels_last',
                activation='relu', border_mode='same')(main_input)
        x = concatenate([x1, x3, x5], axis = 3)
    
    with tf.name_scope('dropout1'): 
        x = Dropout(DROPOUT1)(x)
    
    with tf.name_scope('flatten1'): 
        x = Flatten()(x)
    
    with tf.name_scope('dense1'): 
        x = Dense(DENSESIZE1, activation=DENS_ACT_FN1)(x)
   
    with tf.name_scope('dropout2'):
        x = Dropout(DROPOUT2)(x)
    
    with tf.name_scope('dense2'): 
        x = Dense(DENSESIZE2, activation=DENS_ACT_FN2)(x)
    
    with tf.name_scope('em_trk_none_netout'):
        em_trk_none = Dense(3, activation='softmax', 
                            name='em_trk_none_netout')(x)
    
    with tf.name_scope('michel_netout'):
        michel = Dense(1, activation='sigmoid', name='michel_netout')(x)
    
    with tf.name_scope('sgd'):
        sgd = SGD(lr=0.01, decay=1E-5, momentum=0.9, nesterov=True)
    
    model = Model(inputs=[main_input], outputs=[em_trk_none, michel])
    
    model.compile(optimizer=sgd,
            
                  loss={'em_trk_none_netout' : 'categorical_crossentropy',
                        'michel_netout'      : 'mean_squared_error'},
                  
                  loss_weights={'em_trk_none_netout' : 0.1,
                                'michel_netout'      : 1.0}
                 )

################################################################################
# Read data
# Do one file at a time in order to fit in memory
# Max events at once is roughly 200,000

tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                          write_graph=True, write_images=False)
    
# Build validation sample before training so keep consistency in validation sets
X_test, X_val                     = None, None
em_trk_none_test, em_trk_none_val = None, None
michel_test, michel_val           = None, None

for directory in os.listdir(INPUT_DIR):

    print 'Reading data from', directory
    
    fnameX = [f for f in os.listdir(INPUT_DIR + '/' + directory) 
              if '_x.npy' in f]
    fnameY = [f for f in os.listdir(INPUT_DIR + '/' + directory) 
              if '_y.npy' in f]
    
    if len(fnameX) != 1 or len(fnameY) != 1: continue 
    
    dataX = np.load(INPUT_DIR + '/' + directory + '/' + fnameX[0])
    dataY = np.load(INPUT_DIR + '/' + directory + '/' + fnameY[0])
    if dataX.dtype != np.dtype('float32'): dataX = X_train.astype('float32')
    
    n_patches    = dataX.shape[0]
    n_validation = int(math.floor(FRACTION_VALIDATION * n_patches))
    n_testing    = int(math.floor(FRACTION_TEST * n_patches))
    n_training   = n_patches - n_validation - n_testing
    
    dataX = dataX.reshape(n_patches, PATCH_SIZE_W, PATCH_SIZE_D, 1)
    
    if X_test is None:
        X_test  = dataX[n_training:n_training + n_testing]
        X_val   = dataX[n_training + n_testing:]
        
        em_trk_none_test = dataY[n_training:n_training + n_testing, [0, 1, 3]]
        em_trk_none_val  = dataY[n_training + n_testing:, [0, 1, 3]]
        
        michel_test = dataY[n_training:n_training + n_testing, [2]]
        michel_val  = dataY[n_training + n_testing:, [2]]
    
    else:
        X_test  = np.concatenate((X_test, 
                dataX[n_training:n_training + n_testing]))
        X_val   = np.concatenate((X_val, 
                dataX[n_training + n_testing:]))
        
        em_trk_none_test = np.concatenate((em_trk_none_test,
                dataY[n_training:n_training + n_testing, [0, 1, 3]]))
        em_trk_none_val  = np.concatenate((em_trk_none_val, 
                dataY[n_training + n_testing:, [0, 1, 3]]))
        
        michel_test = np.concatenate((michel_test, 
                dataY[n_training:n_training + n_testing, [2]]))
        michel_val  = np.concatenate((michel_val, 
                dataY[n_training + n_testing:, [2]]))

print ('Test:', X_test.shape, em_trk_none_test.shape, michel_test.shape) 
print ('Val:', X_val.shape, em_trk_none_val.shape, michel_val.shape) 
quit()
    
for epoch in range(N_EPOCH):

    for directory in os.listdir(INPUT_DIR):
    
        print 'Reading data from', directory
        
        fnameX = [f for f in os.listdir(INPUT_DIR + '/' + directory) 
                  if '_x.npy' in f]
        fnameY = [f for f in os.listdir(INPUT_DIR + '/' + directory) 
                  if '_y.npy' in f]
        
        if len(fnameX) != 1 or len(fnameY) != 1: continue 
        
        dataX = np.load(INPUT_DIR + '/' + directory + '/' + fnameX[0])
        dataY = np.load(INPUT_DIR + '/' + directory + '/' + fnameY[0])
        if dataX.dtype != np.dtype('float32'): dataX = X_train.astype('float32')
        
        n_patches    = dataX.shape[0]
        n_validation = int(math.floor(FRACTION_VALIDATION * n_patches))
        n_testing    = int(math.floor(FRACTION_TEST * n_patches))
        n_training   = n_patches - n_validation - n_testing
        
        dataX = dataX.reshape(n_patches, PATCH_SIZE_W, PATCH_SIZE_D, 1)
        
        X_train           = dataX[:n_training]
        em_trk_none_train = dataY[:n_training, [0, 1, 3]]
        michel_train      = dataY[:n_training, [2]]
        
        print ('total', n_patches)
        print ('Train:', X_train.shape, em_trk_none_train.shape, 
                michel_train.shape)
        print 'Patch size: ', PATCH_SIZE_W, 'x', PATCH_SIZE_D
    
        ########################################################################
        # Do training
        print 'Configuration', CONFIG_NAME
        
        
        h = model.fit( {'main_input': X_train},
                       {'em_trk_none_netout': em_trk_none_train,
                        'michel_netout': michel_train},
                        
                       validation_data=(
                          {'main_input': X_val},
                          {'em_trk_none_netout': em_trk_none_val,
                          'michel_netout': michel_val}),
                          
                       callbacks=[tb_callback],
                        
                       batch_size=BATCH_SIZE, 
                       epochs=1, 
                       shuffle=True, 
                       verbose=1
                       
                    )
        
        
score = model.evaluate({'main_input': X_test}, 
                       {'em_trk_none_netout': em_trk_none_test, 
                       'michel_netout': michel_test},
                       verbose=0)
        
print('Test score:', score)

X_train           = None
em_trk_none_train = None
michel_train      = None
        
X_test           = None
em_trk_none_test = None
michel_test      = None

################################################################################
# Save model
if save_model(model, args.output + '/' + CONFIG_NAME):
    print ('Finished')
else:
    print ('Error, couldn\'t save model')
