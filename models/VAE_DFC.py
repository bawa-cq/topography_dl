
import keras
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.applications.vgg19 import VGG19

from losses import VAE_DFC_loss
from models.VAE import build_VAE

def build_VAE_DFC(input, filters=[64, 128, 128, 256], kernel_down=4, kernel_up=3, latent_dim=100, selected_VGG_layer_weights=[1.0, 0.75, 0.5, 0.5], selected_VGG_layers = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2']):

    assert len(selected_VGG_layers) == len(selected_VGG_layer_weights)

    VAE, encoder, decoder = build_VAE(input, filters, kernel_down, kernel_up, latent_dim)

    z_mean, z_log_var, z = encoder(input)
    output = decoder(z)

    #load perceptual loss model: vgg19 layers and give weight
    y = int(input.shape[1])
    x = int(input.shape[2])
    perceptual_model = VGG19(include_top=False, weights='imagenet', input_shape=(x, y, 3))
    outputs_VGG = [perceptual_model.get_layer(l).output for l in selected_VGG_layers]

    lossModel = Model(perceptual_model.input, outputs_VGG)
    for layer in lossModel.layers:
        layer.trainable=False


    weight = K.variable(0.0)
    VAE.add_loss(VAE_DFC_loss(input, output, z_log_var, z_mean, weight, lossModel, selected_VGG_layer_weights))

    VAE.compile(optimizer = Adam(lr=1e-6, clipnorm=1))

    return VAE, encoder, decoder, lossModel
