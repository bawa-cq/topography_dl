import numpy as np
import math
import tensorflow as tf
import os
import datetime

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec

from sklearn.model_selection import train_test_split

from callbacks import AnnealingCallback

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def pre_process(data_in, x=96, y=96, altitude_min=-10880.587890625, altitude_max=8613.15625, test_size=0.05, nbr_rotations=1):
    dim1 = data_in.shape[1]
    dim2 = data_in.shape[0]
    altitude_min = np.abs(altitude_min)
    altitude_max = np.abs(altitude_max)

    data_in = (data_in + altitude_min) / (altitude_max + altitude_min)
    # is now in [0,1], where 0 is altitude_min, 1 is altitude_max

    num_images = int(dim1*dim2 / (x*y))
    cutoff_x = math.ceil(x * (dim1/x-int(dim1/x)))
    cutoff_y = math.ceil(y * (dim2/y-int(dim2/y)))

    # transform into num_images samples of size y by x
    data = np.zeros([num_images, y, x])
    i=0
    for r in range(0, dim2-cutoff_y, y):
        for c in range(0, dim1-cutoff_x, x):
            data[i, :, :] = data_in[r:r+y, c:c+x]
            i = i + 1

    #train - test split
    X_training, X_test = train_test_split(data, test_size=test_size, random_state=7)

    # data augmentation: add rotations
    index = X_training.shape[0]

    if nbr_rotations == 3:
        X_train = np.zeros([index*4, y, x])
    elif nbr_rotations == 2:
        X_train = np.zeros([index*3, y, x])
    elif nbr_rotations == 1:
        X_train = np.zeros([index*2, y, x])
    else:
        X_train = np.zeros_like(X_training)

    X_train[0:index, :, :] = X_training

    if nbr_rotations > 0:
        for j in range(index):
            X_train[index*1 + j, :, :] = X_training[j, :, :].T
            if nbr_rotations == 3:
                X_train[index*2 + j, :, :] = np.flip(X_training[j, :, :])
                X_train[index*3 + j, :, :] = np.flip(X_training[j, :, :].T)
            elif nbr_rotations == 2:
                X_train[index*2 + j, :, :] = np.flip(X_training[j, :, :])

    #train - val train_test_split
    X_train, X_val = train_test_split(X_train, test_size=test_size, random_state=7)

    #reshaping for keras model input
    X_train = X_train.reshape(-1, y, x, 1)
    X_val   = X_val.reshape(-1, y, x, 1)
    X_test  = X_test.reshape(-1, y, x, 1)

    return X_train, X_val, X_test

def set_lr_schedule(learning_rate):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    def lr_schedule(epoch):

        if epoch%2 == 0:
            learning_rate = learning_rate / (1.0005*(epoch+1))

        return learning_rate

    #return the function
    return lr_schedule

def get_callbacks(KL_annealing=True, save_weights=True, TensorBoard=False, ReduceLROnPlateau=False, LearningRateScheduler=False, lr=1e-5):
    callback_list = []

    #use exponential annealing rate for KL divergence loss
    if KL_annealing:
        weight = K.variable(0.0)
        callback_list.append(AnnealingCallback(weight))

    #save best weights
    if save_weights:
        if not(os.path.exists('model_weights')):
            os.mkdir('model_weights')

        path = 'model_weights/' + datetime.datetime.now().strftime('%Y%m%d')
        if not(os.path.exists(path)):
            os.mkdir(path)
        else:
            filepath = path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            save_weights_callback = keras.callbacks.ModelCheckpoint(filepath, save_best_only=True,
                                                                          save_weights_only=True)
            callback_list.append(save_weights_callback)

    #TensorBoard
    if TensorBoard:
        if not(os.path.exists('logs')):
            os.mkdir('logs')

        if not(os.path.exists('logs/fit')):
            os.mkdir('logs/fit')
        else:
            log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_grads=True, write_images=True)
            callback_list.append(tensorboard_callback)

    if ReduceLROnPlateau:
        callback_list.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-6))

    if LearningRateScheduler:
        callback_list.append(keras.callbacks.LearningRateScheduler(set_lr_schedule(lr), verbose=1))

    return callback_list
