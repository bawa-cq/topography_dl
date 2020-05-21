import keras
from keras import backend as K

def charbonnier(y_true, y_pred):
    return K.mean(K.sqrt(K.square(y_true - y_pred) + 1e-10))

def charbonnier_sum(y_true, y_pred):
    return K.sum(K.sqrt(K.square(y_true - y_pred) + 1e-10), axis=-1)

def KL_loss(z_mean, z_log_var):
    loss  = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    loss  = K.sum(loss, axis=-1)
    loss *= -0.5

    return loss

def perceptual_loss(y_true, y_pred, lossModel, selected_VGG_layer_weights):
    y_true = keras.layers.Concatenate()([y_true, y_true, y_true])
    y_pred = keras.layers.Concatenate()([y_pred, y_pred, y_pred])

    h1_list = lossModel(y_true)
    h2_list = lossModel(y_pred)

    rc_loss = 0.0

    for h1, h2, weight in zip(h1_list, h2_list, selected_VGG_layer_weights):
        h1 = K.batch_flatten(h1)
        h2 = K.batch_flatten(h2)
        #rc_loss = rc_loss + weight * K.mean(K.sum(K.square(h1 - h2), axis=-1))
        rc_loss = rc_loss + weight * charbonnier_sum(h1, h2)
    return 0.5*rc_loss

def VAE_loss(y_true, y_pred, z_log_var, z_mean, weight):
    rc_loss = charbonnier_sum(y_true, y_pred)
    #rc_loss = charbonnier(y_true, y_pred)
    #rc_loss = K.sum(K.square(y_true - y_pred), axis=-1)

    return K.mean(rc_loss + weight*KL_loss(z_mean, z_log_var))

def VAE_DFC_loss(y_true, y_pred, z_log_var, z_mean, weight, lossModel, selected_VGG_layer_weights):
    PM_loss = perceptual_loss(y_true, y_pred, lossModel, selected_VGG_layer_weights)

    return K.mean(PM_loss + weight*KL_loss(z_mean, z_log_var), axis=-1)
