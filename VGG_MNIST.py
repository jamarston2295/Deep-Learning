import numpy as np
import pandas as pd

import keras,os
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import array_to_img
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt
%matplotlib inline

df_train = pd.read_csv('/content/drive/My Drive/train.csv')
df_test  = pd.read_csv('/content/drive/My Drive/test.csv')

y_train_temp = to_categorical(df_train.iloc[:, 0].values, num_classes=10)
x_train_temp = df_train.iloc[:, 1:].values.reshape(-1, 28, 28, 1)

permut = np.random.permutation(x_train_temp.shape[0])

x_train = x_train_temp[permut]
y_train = y_train_temp[permut]

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu')) #added line
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #added line
model.add(MaxPooling2D())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) #added line
model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) #added line
model.add(MaxPooling2D()) #added line

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

adam = Adam(lr=1e-4, decay=1e-6)
model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])

!mkdir models
!mkdir logs

tensorboard = TensorBoard(write_grads=True, write_images=True)
chkpoint = ModelCheckpoint("models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)
hist = model.fit(x_train, y_train, epochs=20, callbacks=[tensorboard, chkpoint], validation_split=0.2)

best_model = model

x_test = df_test.values.reshape(-1, 28, 28, 1)
print('Test set: {}'.format(x_test.shape))

probs = best_model.predict(x_test, verbose=1)
preds = np.argmax(probs, axis=1)

print("Predictions: {}".format(preds.shape))

plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
