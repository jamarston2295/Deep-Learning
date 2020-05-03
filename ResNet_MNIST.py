from __future__ import print_function
import keras, os
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_classes = 10

train = pd.read_csv('/content/drive/My Drive/train.csv')
test = pd.read_csv('/content/drive/My Drive/test.csv')

# ResNet-20
depth = 20

# TRAINING

# # drop training label
# labels = train['label'].values
# train.drop('label', axis=1, inplace=True)

# # reshape
# images = train.values
# images = np.array([np.reshape(i, (28, 28)) for i in images])
# images = np.array([i.flatten() for i in images])

# label_binarizer = LabelBinarizer()
# labels = label_binarizer.fit_transform(labels)

# # splitting
# x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

# input image dimensions
input_shape = (28,28,1)

# normalizing training and test data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

def learn_rate_schedule(epoch): # not really needed until higher epochs which I can't use yet
    """
    Learning schedule so learning rate decreases after 80, 120, 160, 180 epochs
    Arguments
        epoch: number of epochs
    Returns
        learn_rate: learning rate
    """

    learn_rate = 1e-3
    if epoch > 180:
        learn_rate *= 0.5e-3
    elif epoch > 160:
        learn_rate *= 1e-3
    elif epoch > 120:
        learn_rate *= 1e-2
    elif epoch > 80:
        learn_rate *= 1e-1

    print('Learning rate: ', learn_rate)

    return learn_rate


def resnet_layer(inputs, num_filters=14, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, Padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    x = inputs

    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)

    return x

def resnet_model(input_shape, depth, num_classes):
    num_filters = 14
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # for first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # for first layer but not first stack
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # add classifier on top
    x = AveragePooling2D(pool_size=(2,2))(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # instantiate model
    model = Model(inputs=inputs, outputs=outputs)

    return model


def model_runner(batch_size, epochs):
    """
    Runs the model
    Arguments
        batch_size: the batch size
        epoch: number of epochs
    """

    model = resnet_model(input_shape=input_shape, depth=depth, num_classes=10)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learn_rate_schedule(0)), metrics=['accuracy'])
    # model.summary()

    learn_rate_scheduler = LearningRateScheduler(learn_rate_schedule)

    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[learn_rate_scheduler])

    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show()

if __name__ == '__main__':

    model_runner(batch_size=32, epochs=20)
