'''
Convolutional Neural Networks
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from app.main.base.run import X_train,X_validate,y_train,y_validate,X_test,y_test
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout




# import random
# #show some clothing
# image = X_train[random.randint(1, 500),:].reshape((28,28))
# plt.imshow(image)
# plt.show()
# plt.close()

print("\n--------------------------------CNN processing----------------------------------\n")


# A number of values that will be used below are set here.
rows = 28
cols = 28
batch_size = 512
shape = (rows, cols, 1)
num_classes = 10

#reshape the arrays as needed. apply features scaling.
X_train = X_train.reshape(X_train.shape[0], *shape)
X_validate =X_validate.reshape(X_validate.shape[0], *shape)
X_test = X_test.reshape(X_test.shape[0], *shape)


''' highest accuracy CNN'''


# #create cnn model with params:
# #times to repeat convoluting the data and maxPooling.
def createCNN2( filterSize, kernelSize, strides, poolSize, density, epochs):
    model = Sequential()
    # For the convolutional stage, add the number of feature detectors, alongside the feature detector size. 3 indicates to a 3x3 matrix. and set the activation to relu.
    # 32 is a classic filter number.
    model.add(Conv2D(filters=filterSize, kernel_size=kernelSize,padding = 'same', activation='relu', input_shape=shape))
    # Apply the max pooling feature to our feature map. This helps with identifying certain features in different positions within the image, eg. when an animal is facing right or left.
    # 2x2 matrix with a strides of 2
    model.add(MaxPooling2D(pool_size=poolSize, strides=strides))
    model.add(Dropout(0.2))
    # flatten the previous steps into a 1 dimensional vector.
    model.add(Flatten())
    # add the fully connected neuron layers
    model.add(Dense(units=density, activation='relu'))
    # output layer
    # As we are using multiple classification, the softmax feature is used.for Also we have 10 possible outputs, so the the density is 10.
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])# metrics = What you want to maximise

    model.summary()

    # train the model. # epochs=umber of iterations(backpropogations)
    history=model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_validate, y_validate))

    print("\n-----------------------------CNN model accuracy (Validation test) ----------------------------------")

    score = model.evaluate(X_validate, y_validate, verbose=0)

    print('test loss: {:.2f}'.format(score[0]))
    print('test acc: {:.2f}'.format(score[1]))

    # Plot training & validation loss values

    plt.plot(history.history['loss'])
    plt.title('Model loss/accuracy for Validation data')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Loss'], loc='upper left')

    plt2 = plt.twinx()
    color = 'red'
    plt2.plot(history.history['accuracy'], color=color)
    plt.ylabel('Accuracy')
    plt2.legend(['Accuracy'], loc='upper center')
    plt.show()

    print("\n-----------------------------CNN model accuracy (Testing data) ----------------------------------")

    score = model.evaluate(X_test, y_test, verbose=0)

    print('test loss: {:.2f}'.format(score[0]))
    print('test acc: {:.2f}'.format(score[1]))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss/accuracy for Testing data')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Loss'], loc='upper left')
    plt2 = plt.twinx()
    color = 'red'
    plt2.plot(history.history['accuracy'], color=color)
    plt.ylabel('Accuracy')
    plt2.legend(['Accuracy'], loc='upper center')
    # plot accuracy

    plt.show()
    # code adapted from
    # Brownlee, J., 2021. DeepLearning CNN for Fashion - MNIST Clothing Classification.[online] Machine Learning Mastery.Available at: <
    # https: // machinelearningmastery.com / how - to - develop - a - cnn -
    # from -scratch - for -fashion - mnist - clothing - classification / > [Accessed 22 December 2021].

    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')


    plt.show()

    # save model
    model.save('cnn.h5')


# #play around
# # createCNN2( 64, 3, 2, 2, 128, 48)
# # -----------------------------CNN model accuracy with sample ----------------------------------
# # test loss: 0.35
# # test acc: 0.90
# # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
#
# #  -----------------------------CNN model accuracy with actual data----------------------------------
# # test loss: 0.32
# # test acc: 0.92
#
# # createCNN2( 64, 3, 2, 2, 128, 25)
# createCNN2( 64, 3, 2, 2, 128, 19)
# #  -----------------------------CNN model accuracy with actual data----------------------------------
# # test loss: 0.25
# # test acc: 0.92
# print('\n---------------')
# createCNN2( 32,3,2,2,128,10)
#
#
# print('\n---------------')
# createCNN2( 64,3,2,2,128,10)
#
# print('\n---------------')
# createCNN2( 64,3,2,2,128,48)
#
# print('\n---------------')
# createCNN2( 64,3,2,2,128,25)
#
# print('\n---------------')
# createCNN2( 32,3,2,2,128,19)

# best accuracy
print('\n---------------')
createCNN2( 32,3,2,2,128,48)

# '''
# New way without function
# '''
# # datagen = ImageDataGenerator(
# #     featurewise_center=False,  # set input mean to 0 over the dataset
# #     samplewise_center=False,  # set each sample mean to 0
# #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
# #     samplewise_std_normalization=False,  # divide each input by its std
# #     zca_whitening=False,  # apply ZCA whitening
# #     #zca_epsilon=1e-06,  # epsilon for ZCA whitening
# #     rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
# #     # randomly shift images horizontally (fraction of total width)
# #     width_shift_range=0.1,
# #     # randomly shift images vertically (fraction of total height)
# #     height_shift_range=0.1,
# #     shear_range=0.,  # set range for random shear
# #     zoom_range=0.,  # set range for random zoom
# #     channel_shift_range=0.,  # set range for random channel shifts
# #     # set mode for filling points outside the input boundaries
# #     fill_mode='nearest',
# #     cval=0.,  # value used for fill_mode = "constant"
# #     horizontal_flip=True,  # randomly flip images
# #     vertical_flip=False,  # randomly flip images
# #     # set rescaling factor (applied before any other transformation)
# #     rescale=None,
# #     # set function that will be applied on each input
# #     preprocessing_function=None,
# #     # image data format, either "channels_first" or "channels_last"
# #     data_format=None,
# #     # fraction of images reserved for validation (strictly between 0 and 1)
# #     validation_split=0.0)
# #
# # datagen.fit(X_train)
# #
# # model = Sequential()
# # # kernals pick out what you need (edges,corners)
# # model.add(Conv2D(64, kernel_size=4, activation='relu', strides=1, padding='same', input_shape=shape))
# # model.add(Conv2D(32, 3, activation='relu'))
# # model.add(MaxPooling2D(pool_size=2))
# # model.add(Conv2D(64, 3, activation='relu', padding='same'))
# # model.add(Conv2D(64, 3, activation='relu'))
# # model.add(MaxPooling2D(pool_size=2))
# # model.add(Flatten())
# # model.add(Dense(128, activation='relu'))
# # # last layer should be number of classes
# # model.add(Dense(num_classes))
# # model.add(Activation('softmax'))
# #
# #
# # custom = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
# #
# # model.compile(loss='categorical_crossentropy',
# #               optimizer=custom,
# #               metrics=['accuracy'])
# #
# # #model.summary()
# #
# # history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
# #                     epochs=48)
# #
# # #make predictions (will give a probability distribution)
# # pred = model.predict(X_test)
# # #now pick the most likely outcome
# # pred = np.argmax(pred,axis=1)
# # y_compare = np.argmax(y_test,axis=1)
# # #and calculate accuracy
# # score = metrics.accuracy_score(y_compare, pred)
# # print("Accuracy score: {}".format(score))
# #
# # # Plot training & validation loss values
# # print(history.history.keys())
# # plt.plot(history.history['loss'])
# # plt.title('Model loss/accuracy')
# # plt.ylabel('Loss')
# # plt.xlabel('Epoch')
# # plt.legend(['Loss'], loc='upper left')
# #
# # plt2=plt.twinx()
# # color = 'red'
# # plt2.plot(history.history['accuracy'],color=color)
# # plt.ylabel('Accuracy')
# # plt2.legend(['Accuracy'], loc='upper center')
# # plt.show()
# #
# # # add confusion confusion_matrix
# #
# # # # Save model and weights
# # # model_path = os.path.join(save_dir, model_name)
# # # model.save(model_path)
# # # print('Saved trained model at %s ' % model_path)
