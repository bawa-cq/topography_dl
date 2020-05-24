import numpy as np
import datetime

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import keras
from keras.layers import Input

from callbacks import AnnealingCallback

from utils import pre_process
from utils import build_model
from utils import get_callbacks


if __name__ == '__main__':

    nc = Dataset('../Data/tohoku_2020.nc', 'r')
    Z = np.flip(np.array(nc.variables['elevation']), axis=0)

    X_train, X_val, X_test = pre_process(Z)

    #build model
    inputs = Input(shape = (y, x, 1), name='encoder_inputs')

    # type:
    #'AE'=autoencoder
    #'VAE'=variational autoencoder
    #'DFC_VAE'=deep feature consistent variational autoencoder
    model, encoder, decoder, lossModel, encoded = build_model(type='AE')

    #hyper parameters
    epochs = 2
    batch_size = 128
    learning_rate = 1e-6

    optimizer = keras.optimizers.Adam(lr=learning_rate, clipnorm=1)

    #callbacks
    callback_list = get_callbacks()

    model.compile(optimizer = optimizer)

    model.fit(X_train, batch_size=batch_size, epochs=epochs, verbose=1, #initial_epoch=,
          validation_data=(X_val, None),
          callbacks=callback_list
         )

    model_train = model.history
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']

    plt.figure()
    plt.plot(range(epochs), loss, 'bo', label='Training loss')
    plt.plot(range(epochs), val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()
