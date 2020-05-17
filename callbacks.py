from keras.callbacks import Callback
from keras import backend as K

#exponential annealing rate for KL divergence loss
class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight
    def on_epoch_end(self, epoch, logs={}):
        new_weight = K.get_value(self.weight)

        if new_weight!=1:
            if epoch+1 >= KL_start:
                print("KL_weight: " + str(new_weight))
                new_weight = sigmoid(-10 + KL_growth*(epoch+1-KL_start))

        if new_weight > 1 - 1e-6:
            new_weight = 1

        K.set_value(self.weight, new_weight)
