
import keras
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
from keras import backend as K

from utils import ReflectionPadding2D, sampling
from losses import VAE_loss

def build_VAE(input, filters=[64, 128, 128, 256], kernel_down=4, kernel_up=3, latent_dim=100):
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

    #shape retrieval for decoder part
    shape = K.int_shape(a)

    # generate latent vector Q(z|X)
    a = Flatten()(a)
    a = Dense(latent_dim, activation='relu', kernel_initializer='he_uniform')(a)

    z_mean = Dense(latent_dim, name='z_mean')(a)
    z_log_var = Dense(latent_dim, name='z_log_var')(a)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    #instatiate encoder model
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')

    #decoder network
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

    a = Dense(latent_dim, activation='relu', kernel_initializer='he_uniform')(latent_inputs)
    a = Dense(shape[1] * shape[2] * shape[3], activation='relu', kernel_initializer='he_uniform')(a)
    a = Reshape((shape[1], shape[2], shape[3]))(a)

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

    # instantiate decoder model
    decoder = Model(latent_inputs, out, name='decoder')

    output = decoder(encoder(input)[2])
    VAE = Model(input, output, name='VAE')

    weight = K.variable(0.0)
    VAE.add_loss(VAE_loss(K.flatten(input), K.flatten(output), z_log_var, z_mean, weight))

    VAE.compile(optimizer = Adam(lr=1e-6, clipnorm=1))

    return VAE, encoder, decoder
