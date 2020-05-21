
import keras
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.layers import Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam

from utils import ReflectionPadding2D
from losses import charbonnier

def build_AE(input, filters=[64, 128, 128, 256], kernel_down=4, kernel_up=3):
    a = input

    #encoder network
    for i in range(3):
        a = ReflectionPadding2D(padding=(1,1))(a)
        a = Conv2D(filters[i], kernel_initializer='he_uniform',
                   kernel_size=kernel_down, #kernel_regularizer=regularizers.l1(1e-4),
                   strides=2, padding='valid')(a)
        a = BatchNormalization(axis=3)(a)
        a = keras.layers.LeakyReLU()(a)

    a = ReflectionPadding2D(padding=(1,1))(a)
    a = Conv2D(filters[i+1], kernel_initializer='he_uniform',
               kernel_size=kernel_down, #kernel_regularizer=regularizers.l1(1e-4),
               strides=2, padding='valid')(a)
    a = BatchNormalization(axis=3)(a)
    a = keras.layers.LeakyReLU(name='encoded')(a)

    #decoder network
    filters.reverse()
    for i in range(3):

        a = UpSampling2D((2,2))(a)
        a = ReflectionPadding2D(padding=(1,1))(a)
        a = Conv2D(filters[i+1], kernel_size=kernel_up, #kernel_regularizer=regularizers.l2(1e-3),
                   strides=1, kernel_initializer='he_uniform', padding='valid')(a)
        a = BatchNormalization(axis=3)(a)
        a = keras.layers.LeakyReLU()(a)

    a = UpSampling2D((2,2))(a)
    a = ReflectionPadding2D(padding=(1,1))(a)
    out = Conv2D(filters=1, kernel_size=kernel_up, #kernel_regularizer=regularizers.l2(1e-3),
                 kernel_initializer='he_normal', activation='sigmoid',
                 padding='valid', name='decoder_output')(a)

    AE = Model(input, out, name='AE')
    AE.add_loss(charbonnier(input, out))

    AE.compile(optimizer = Adam(lr=1e-2))

    return AE
