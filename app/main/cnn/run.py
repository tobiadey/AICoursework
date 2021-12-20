'''
Convolutional Neural Networks
'''

import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
import io
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# import seaborn as sns
from app.main.base.run import X,y
# import svm accuracy_score

#use the keras built in to ensure the targets are categories
y = tensorflow.keras.utils.to_categorical(y)
#and check this...
print(y[:5])


import random
#show some clothing
image = X[random.randint(1, 500),:].reshape((28,28))
plt.imshow(image)
plt.show()
plt.close()

'''
split the data, and expand dimensions so that the input to the CNN is a 3d tensor.
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# A number of values that will be used below are set here.

rows = 28
cols = 28
batch_size = 512
shape = (rows, cols, 1)
num_classes = 10

X_train = X_train.reshape(X_train.shape[0], *shape)
X_test =X_test.reshape(X_test.shape[0], *shape)


print(X_train.shape)
print(X_train[1][1])


'''
New way
'''
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    #zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

datagen.fit(X_train)


def createCNN(times,fs, ks,strides,poolSize,density,epochs):
    model = Sequential()
    model.add(Conv2D(fs, kernel_size=ks, activation='relu', strides=strides, padding='same', input_shape=shape))
    for i in range (times):
        # kernals pick out what you need (edges,corners)
        model.add(Conv2D(fs, ks, activation='relu'))
        model.add(MaxPooling2D(pool_size=poolSize))
    model.add(Flatten())
    model.add(Dense(density, activation='relu'))
    # last layer should be number of classes
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    custom = tensorflow.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy',
              optimizer=custom,
              metrics=['accuracy'])

#model.summary()

    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs)

#make predictions (will give a probability distribution)
    pred = model.predict(X_test)
#now pick the most likely outcome
    pred = np.argmax(pred,axis=1)
    y_compare = np.argmax(y_test,axis=1)
#and calculate accuracy
    score = metrics.accuracy_score(y_compare, pred)
    print("Accuracy score: {}".format(score))

# Plot training & validation loss values
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.title('Model loss/accuracy')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Loss'], loc='upper left')

    plt2=plt.twinx()
    color = 'red'
    plt2.plot(history.history['accuracy'],color=color)
    plt.ylabel('Accuracy')
    plt2.legend(['Accuracy'], loc='upper center')
    plt.show()

createCNN(2,32,4,2,2,128,10)


'''old way'''


# cnn_model = Sequential([
#     Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape),
#     MaxPooling2D(pool_size=2),
#     Dropout(0.2),
#     Flatten(),
#     Dense(32, activation='relu'),
#     Dense(10, activation='softmax')
# ])


# modelOld = Sequential()
# modelOld.add(Conv2D(32, kernel_size=3, strides=1, input_shape=im_shape))
# modelOld.add(Activation('relu'))
# modelOld.add(MaxPooling2D(pool_size=2))
# modelOld.add(Flatten())
# modelOld.add(Dense(128))
# modelOld.add(Activation('relu'))
# modelOld.add(Dense(num_classes))
# modelOld.add(Activation('softmax'))
# modelOld.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# modelOld.summary()
#
# history = modelOld.fit(X_train,y_train,verbose=2,epochs=24)
#
# #make predictions (will give a probability distribution)
# pred = modelOld.predict(X_test)
# #now pick the most likely outcome
# pred = np.argmax(pred,axis=1)
# y_compare = np.argmax(y_test,axis=1)
# #and calculate accuracy
# score = metrics.accuracy_score(y_compare, pred)
# print("Accuracy score: {}".format(score))
#
# '''
# this model very quickly fits to the training data giving 100% accuracy, and the loss has dropped to a low value.
# but the accuracy result isn't perhaps as good as we might like.
# let's add some additional convolution layers,the idea being to try and discover more features.
#
# '''
# modelOld2 = Sequential()
# modelOld2.add(Conv2D(64, kernel_size=4, activation='relu', strides=1, padding='same', input_shape= im_shape))
# modelOld2.add(Conv2D(32, 3, activation='relu'))
# modelOld2.add(MaxPooling2D(pool_size=2))
# modelOld2.add(Conv2D(64, 3, activation='relu', padding='same'))
# modelOld2.add(Conv2D(64, 3, activation='relu'))
# modelOld2.add(MaxPooling2D(pool_size= 2))
#
# modelOld2.add(Flatten())
# modelOld2.add(Dense(128, activation='relu'))
#
# modelOld2.add(Dense(num_classes))
# modelOld2.add(Activation('softmax'))
#
# modelOld2.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# modelOld2.summary()
#
# # Running this model for 16 epochs appears to lead to stability in both loss and accuracy over the training set.
# # Notice that with the additional layers, training is slower
# # (despite there being fewer parameters to tune, owing to the second pooling step).
# modelOld2.fit(X_train,y_train,verbose=2,epochs=16)
#
# #make predictions (will give a probability distribution)
# pred = modelOld2.predict(X_test)
# #now pick the most likely outcome
# pred = np.argmax(pred,axis=1)
# y_compare = np.argmax(y_test,axis=1)
# #and calculate accuracy
# score = metrics.accuracy_score(y_compare, pred)
# print("Accuracy score: {}".format(score))








# add confusion confusion_matrix

# # Save model and weights
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)

# Questions

# my loss and accuracy stays constant, is this not a problem
# maybe i am doing smt wrong


'''
Convolutional Neural Networks
'''

import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
import io
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# import seaborn as sns
from app.main.base.run import X,y
# import svm accuracy_score

#use the keras built in to ensure the targets are categories
y = tensorflow.keras.utils.to_categorical(y)
#and check this...
print(y[:5])


import random
#show some clothing
image = X[random.randint(1, 500),:].reshape((28,28))
plt.imshow(image)
plt.show()
plt.close()

'''
split the data, and expand dimensions so that the input to the CNN is a 3d tensor.
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# A number of values that will be used below are set here.
# im_rows = 28
# im_cols = 28
# batch_size = 512
# num_classes = y.shape[1]
# im_shape = (im_rows, im_cols, 1)

rows = 28
cols = 28
batch_size = 512
shape = (rows, cols, 1)
num_classes = 10

X_train = X_train.reshape(X_train.shape[0], *shape)
X_test =X_test.reshape(X_test.shape[0], *shape)


print(X_train.shape)
print(X_train[1][1])


# cnn_model = Sequential([
#     Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape),
#     MaxPooling2D(pool_size=2),
#     Dropout(0.2),
#     Flatten(),
#     Dense(32, activation='relu'),
#     Dense(10, activation='softmax')
# ])


# modelOld = Sequential()
# modelOld.add(Conv2D(32, kernel_size=3, strides=1, input_shape=im_shape))
# modelOld.add(Activation('relu'))
# modelOld.add(MaxPooling2D(pool_size=2))
# modelOld.add(Flatten())
# modelOld.add(Dense(128))
# modelOld.add(Activation('relu'))
# modelOld.add(Dense(num_classes))
# modelOld.add(Activation('softmax'))
# modelOld.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# modelOld.summary()
#
# history = modelOld.fit(X_train,y_train,verbose=2,epochs=24)
#
# #make predictions (will give a probability distribution)
# pred = modelOld.predict(X_test)
# #now pick the most likely outcome
# pred = np.argmax(pred,axis=1)
# y_compare = np.argmax(y_test,axis=1)
# #and calculate accuracy
# score = metrics.accuracy_score(y_compare, pred)
# print("Accuracy score: {}".format(score))
#
# '''
# this model very quickly fits to the training data giving 100% accuracy, and the loss has dropped to a low value.
# but the accuracy result isn't perhaps as good as we might like.
# let's add some additional convolution layers,the idea being to try and discover more features.
#
# '''
# modelOld2 = Sequential()
# modelOld2.add(Conv2D(64, kernel_size=4, activation='relu', strides=1, padding='same', input_shape= im_shape))
# modelOld2.add(Conv2D(32, 3, activation='relu'))
# modelOld2.add(MaxPooling2D(pool_size=2))
# modelOld2.add(Conv2D(64, 3, activation='relu', padding='same'))
# modelOld2.add(Conv2D(64, 3, activation='relu'))
# modelOld2.add(MaxPooling2D(pool_size= 2))
#
# modelOld2.add(Flatten())
# modelOld2.add(Dense(128, activation='relu'))
#
# modelOld2.add(Dense(num_classes))
# modelOld2.add(Activation('softmax'))
#
# modelOld2.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# modelOld2.summary()
#
# # Running this model for 16 epochs appears to lead to stability in both loss and accuracy over the training set.
# # Notice that with the additional layers, training is slower
# # (despite there being fewer parameters to tune, owing to the second pooling step).
# modelOld2.fit(X_train,y_train,verbose=2,epochs=16)
#
# #make predictions (will give a probability distribution)
# pred = modelOld2.predict(X_test)
# #now pick the most likely outcome
# pred = np.argmax(pred,axis=1)
# y_compare = np.argmax(y_test,axis=1)
# #and calculate accuracy
# score = metrics.accuracy_score(y_compare, pred)
# print("Accuracy score: {}".format(score))

'''
New way without function
'''
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     #zca_epsilon=1e-06,  # epsilon for ZCA whitening
#     rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
#     # randomly shift images horizontally (fraction of total width)
#     width_shift_range=0.1,
#     # randomly shift images vertically (fraction of total height)
#     height_shift_range=0.1,
#     shear_range=0.,  # set range for random shear
#     zoom_range=0.,  # set range for random zoom
#     channel_shift_range=0.,  # set range for random channel shifts
#     # set mode for filling points outside the input boundaries
#     fill_mode='nearest',
#     cval=0.,  # value used for fill_mode = "constant"
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=False,  # randomly flip images
#     # set rescaling factor (applied before any other transformation)
#     rescale=None,
#     # set function that will be applied on each input
#     preprocessing_function=None,
#     # image data format, either "channels_first" or "channels_last"
#     data_format=None,
#     # fraction of images reserved for validation (strictly between 0 and 1)
#     validation_split=0.0)
#
# datagen.fit(X_train)
#
# model = Sequential()
# # kernals pick out what you need (edges,corners)
# model.add(Conv2D(64, kernel_size=4, activation='relu', strides=1, padding='same', input_shape=shape))
# model.add(Conv2D(32, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(64, 3, activation='relu', padding='same'))
# model.add(Conv2D(64, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# # last layer should be number of classes
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
#
#
# custom = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=custom,
#               metrics=['accuracy'])
#
# #model.summary()
#
# history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
#                     epochs=48)
#
# #make predictions (will give a probability distribution)
# pred = model.predict(X_test)
# #now pick the most likely outcome
# pred = np.argmax(pred,axis=1)
# y_compare = np.argmax(y_test,axis=1)
# #and calculate accuracy
# score = metrics.accuracy_score(y_compare, pred)
# print("Accuracy score: {}".format(score))
#
# # Plot training & validation loss values
# print(history.history.keys())
# plt.plot(history.history['loss'])
# plt.title('Model loss/accuracy')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Loss'], loc='upper left')
#
# plt2=plt.twinx()
# color = 'red'
# plt2.plot(history.history['accuracy'],color=color)
# plt.ylabel('Accuracy')
# plt2.legend(['Accuracy'], loc='upper center')
# plt.show()
#
# # add confusion confusion_matrix
#
# # # Save model and weights
# # model_path = os.path.join(save_dir, model_name)
# # model.save(model_path)
# # print('Saved trained model at %s ' % model_path)
#
# # Questions
#
# # my loss and accuracy stays constant, is this not a problem
# # maybe i am doing smt wrong