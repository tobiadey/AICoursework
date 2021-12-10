
from app.main.base.run import X,y
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#
# print("inside cnn content")
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
#
# from app.main.base.run import *
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator

# X = np.array(X,dtype='float32')
# y = np.array(y,dtype='float32')
# X = X/255.
# # y= y/255.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# X_train = X_train.to_numpy()
# y_train = y_train.to_numpy()
# X_test = X_test.to_numpy()
# y_test = y_test.to_numpy()

print("inside cnn content")
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
# # print(X_test)
# # print(y_train)
# # print(y_test)
#

'''
method 1 few error methods present.
'''
im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)

X_train = X_train.reshape(X_train.shape[0], *im_shape)
X_test =X_test.reshape(X_test.shape[0], *im_shape)
# y_train = y_train.reshape(y_train.shape[0], *im_shape)

print('x_train shape: {}'.format(X_train.shape))
print('x_test shape: {}'.format(X_test.shape))
# print('x_validate shape: {}'.format(y_train.shape))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten

cnn_model = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),

    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])


import tensorflow as tf
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(
    log_dir=r'logs\{}'.format('cnn_1layer'),
    write_graph=True,
    write_grads=True,
    histogram_freq=1,
    write_images=True,
)

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer= tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)


cnn_model.fit(
    X_train, y_train, batch_size=batch_size,
    epochs=10, verbose=1,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard]
)

score = cnn_model.evaluate(X_test, y_test, verbose=0)

print('test loss: {:.4f}'.format(score[0]))
print(' test acc: {:.4f}'.format(score[1]))


'''
p1

'''

# '''
# Tutorial 8 (Introduction to AI)
# Convolutional Neural Networks
# Part 1
#
# CNNs set up a series of convolutions, structurally a series of filters that apply to images.
# The results of applying these is then reduced in size using pooling layers,
# efore the result is passed to one or more standard fully connected feedforward layers.
#
# The network is then trained using labelled data using backpropagation as normal.
# The result of training these deep networks is to learn values for both the standard fully connected
# feedforward layers and for the filters defined in the convolutional layers.
#
#
# '''
# from sklearn.datasets import load_digits
# import tensorflow as tf
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow.keras.utils
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# import io
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
#
# #load the digits dataset from sklearn
# digits = load_digits()
#
# X = digits.images
# y = digits.target
#
# #use the keras built in to ensure the targets are categories
# y = tensorflow.keras.utils.to_categorical(y)
#
# #if you want to check what this looks like, uncomment the below
# # print(X[:5])
# # print(y[:5])
#
# #split into training and testing data
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
#
# #2d matrix
# print(X_train[0][0])
#
# X_train = np.expand_dims(X_train, axis=3)
# X_test = np.expand_dims(X_test, axis=3)
#
# #3d tensor
# print(X_train[0][0])
#
# # Convolutional neural networks can have quite a high number of parameters,
# # so in some cases you might want to gather together values in a definitions section.
# # Here we have just three.
#
# num_classes = y.shape[1]
# save_dir = './'
# model_name = 'ex8Part1_trained_model.h5'
#
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', input_shape= (8,8,1)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.summary()
# history = model.fit(X_train,y_train,verbose=2,epochs=24)
#
# # This has quite quickly fitted itself to the training data, giving 100% accuracy,
# # and the loss is low (if still dropping).
# # Now we use the trained CNN to make predictions, and test accuracy.
# # This is gives our best result yet on this dataset.
#
# #make predictions (will give a probability distribution)
# pred_hot = model.predict(X_test)
# #now pick the most likely outcome
# pred = np.argmax(pred_hot,axis=1)
# y_compare = np.argmax(y_test,axis=1)
# #calculate accuracy
# score = metrics.accuracy_score(y_compare, pred)
#
# print("Accuracy score: {}".format(score))
#
# print(pred_hot[:5])
# print(pred)
#
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train'], loc='upper left')
# plt.show()
#
# mat = confusion_matrix(pred, y_compare)
#
# #using seaborn
# sns.heatmap(mat, square=True, annot=True, cbar=False)
# plt.xlabel('predicted value')
# plt.ylabel('true value');
# plt.show()
#
#
# # Save model and weights
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)


'''
p2
'''

# '''
# Tutorial 8 (Introduction to AI)
# Convolutional Neural Networks
# Part 2
#
# This second example CNN development
# considers how a more powerful model with additional layers might be developed
# and in particular how the training set might be augmented with additional generated data.
#
# The dataset being considered here is the SciKit-Learn Labelled Faces in the Wild,
# which we looked at with PCA and SVM as an exercise in Week 4.
# The model solution gave about 85% accuracy.
#
# '''
#
# import numpy as np
# from sklearn.datasets import fetch_lfw_people
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import tensorflow.keras.utils
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn import metrics
# import io
# import os
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
# '''
# First load the dataset, and take a look at some of it.
# For CNNs we are working with the image data, not the flattened data.
# The target labels are proper names,
# so we need to convert these to one-hot encodings.
# '''
#
# #import that faces dataset
# faces = fetch_lfw_people(min_faces_per_person=60)
# print(faces.target_names)
# print(faces.images.shape)
#
# #set features and target
# X = faces.images
# y = faces.target
#
# #use the keras built in to ensure the targets are categories
# y = tensorflow.keras.utils.to_categorical(y)
# #and check this...
# print(y[:5])
#
# #show some faces
# fig, ax = plt.subplots(3, 5)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap='bone')
#     axi.set(xticks=[], yticks=[],
#             xlabel=faces.target_names[faces.target[i]])
# plt.show()
# plt.close()
#
# '''
# Again split the data, and expand dimensions so that the input to the CNN is a 3d tensor.
# '''
# #split into training and testing data
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
#
# X_train = np.expand_dims(X_train, axis=3)
# X_test = np.expand_dims(X_test, axis=3)
#
# print(X_train.shape)
# #print(X_train[1][1])
#
#
# # A number of values that will be used below are set here.
# batch_size = 128
# num_classes = y.shape[1]
# epochs = 32
# save_dir = './'
# model_name = 'keras_lfw_trained_model.h5'
#
# '''
# For a CNN model, let's start by reusing that from Part 1.
# There's a single convolution layer with 3 x 3 filters, and a stride length of 1.
# Pooling uses the standard 2 x 2 pool size.
#
# The code below compiles the model and trains it for 24 epochs.
# '''
#
# modelOld = Sequential()
# modelOld.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', input_shape= (62, 47, 1)))
# modelOld.add(Activation('relu'))
# modelOld.add(MaxPooling2D(pool_size=(2, 2)))
#
# modelOld.add(Flatten())
# modelOld.add(Dense(128))
# modelOld.add(Activation('relu'))
# modelOld.add(Dense(num_classes))
# modelOld.add(Activation('softmax'))
#
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
#
# '''
# this model very quickly fits to the training data giving 100% accuracy, and the loss has dropped to a low value.
# but the accuracy result isn't perhaps as good as we might like.
# let's add some additional convolution layers,the idea being to try and discover more features.
#
# '''
# modelOld2 = Sequential()
# modelOld2.add(Conv2D(64, kernel_size=(4, 4), activation='relu', strides=1, padding='same', input_shape= X_train[0].shape))
# modelOld2.add(Conv2D(32, (3, 3), activation='relu'))
# modelOld2.add(MaxPooling2D(pool_size=(2, 2)))
# modelOld2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# modelOld2.add(Conv2D(64, (3, 3), activation='relu'))
# modelOld2.add(MaxPooling2D(pool_size=(2, 2)))
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
#
#
# '''
# New way
# '''
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
# model.add(Conv2D(64, kernel_size=(4, 4), activation='relu', strides=1, padding='same', input_shape= X_train[0].shape))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
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
# # Save model and weights
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)
#


'''
no lab work
'''
