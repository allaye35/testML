import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Activation

import fl_dataquest

class MyModel():
    def __init__(self, input_shape, nbclasses=10):
        model = Sequential()
        model.add(Input(input_shape))   # (28,28)
        model.add(Flatten())            # -> vecteur 784
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dense(nbclasses))
        model.add(Activation("softmax"))

        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def fit_it(self, trains, epochs, tests, verbose):
        self.model.fit(trains, epochs=epochs, validation_data=tests, verbose=verbose)

    def evaluate(self, tests, verbose=0):
        return self.model.evaluate(tests, verbose=verbose)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def summary(self):
        self.model.summary()

    def pretty_print_layers(self):
        for layer_i in range(len(self.model.layers)):
            wts = self.model.layers[layer_i].get_weights()
            if len(wts) != 0:
                w, b = wts
                print("Layer", layer_i, "weights shape=", w.shape, "bias shape=", b.shape)

if __name__=='__main__':
    # Petit test local
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(verbose=1)
    dtt, dts = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=1)

    m = MyModel(input_shape, nbclasses=10)
    m.fit_it(trains=dtt, epochs=1, tests=dts, verbose=1)
    loss, acc = m.evaluate(dts, verbose=1)
    print("Test Loss:", loss, " Test Acc:", acc)
