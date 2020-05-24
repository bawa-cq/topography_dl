import numpy as np

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import keras
from keras.layers import Input

from models.AE import build_AE
from models.VAE import build_VAE
from models.VAE_DFC import build_DFC_VAE

from utils import pre_process
from utils import get_callbacks

def build_model(type, kernel_down=4, kernel_up=3, filters=[64, 128, 128, 256], latent_dim =100, selected_VGG_layer_weights=[1.0, 0.75, 0.5, 0.5], selected_VGG_layers=['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2']):

    if type == 'AE':
        model = build_AE(inputs, filters, kernel_down, kernel_up)
        encoder, decoder, lossModel = None, None, None
        encoded = keras.models.Model(inputs=inputs, outputs=model.get_layer('encoded').output)
    elif type == 'VAE':
        model, encoder, decoder = build_VAE(inputs, filters, kernel_down, kernel_up, latent_dim)
        lossModel = None
        encoded = keras.models.Model(inputs=inputs, outputs=model.get_layer('encoder').get_layer('encoded').output)
    elif type == 'DFC_VAE':
        model, encoder, decoder, lossModel = build_DFC_VAE(inputs, filters, kernel_down, kernel_up, latent_dim, selected_VGG_layer_weights, selected_VGG_layers)
        encoded = keras.models.Model(inputs=inputs, outputs=model.get_layer('encoder').get_layer('encoded').output)

    return model, encoder, decoder, lossModel, encoded


if __name__ == '__main__':

    nc = Dataset('../Data/tohoku_2020.nc', 'r')
    Z = np.flip(np.array(nc.variables['elevation']), axis=0)

    x = 96
    y = 96
    X_train, X_val, X_test = pre_process(Z, x, y)

    #build model
    inputs = Input(shape = (y, x, 1), name='encoder_inputs')

    # type:
    #'AE'=autoencoder
    #'VAE'=variational autoencoder
    #'DFC_VAE'=deep feature consistent variational autoencoder
    model, encoder, decoder, lossModel, encoded = build_model(type='DFC_VAE')

    #hyper parameters
    epochs = 50
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
