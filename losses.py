import keras
from keras import backend as K

def charbonnier(y_true, y_pred):
    return K.mean(K.sqrt(K.square(y_true - y_pred) + 1e-10))

def charbonnier_sum(y_true, y_pred):
    return K.sum(K.sqrt(K.square(y_true - y_pred) + 1e-10), axis=-1)

def KL_loss():
    loss  = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    loss  = K.sum(loss, axis=-1)
    loss *= -0.5

    return loss

def perceptual_loss(y_true, y_pred):
    y_true = keras.layers.Concatenate()([y_true, y_true, y_true])
    y_pred = keras.layers.Concatenate()([y_pred, y_pred, y_pred])

    h1_list = lossModel(y_true)
    h2_list = lossModel(y_pred)

    rc_loss = 0.0

    for h1, h2, weight in zip(h1_list, h2_list, selected_pm_layer_weights):
        h1 = K.batch_flatten(h1)
        h2 = K.batch_flatten(h2)
        #rc_loss = rc_loss + weight * K.mean(K.sum(K.square(h1 - h2), axis=-1))
        rc_loss = rc_loss + weight * charbonnier_sum(h1, h2)
    return 0.5*rc_loss

def VAE_loss(y_true, y_pred):
    global weight

    rc_loss = charbonnier_sum(y_true, y_pred)
    #rc_loss = charbonnier(y_true, y_pred)
    #rc_loss = K.sum(K.square(y_true - y_pred), axis=-1)

    KL_loss  = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    KL_loss  = K.sum(KL_loss, axis=-1)
    KL_loss *= -0.5

    return K.mean(rc_loss + weight*KL_loss)

def VAE_DFC_loss(y_true, y_pred):
    global weight

    return K.mean(perceptual_loss(y_true, y_pred) + weight*KL_loss(), axis=-1)
