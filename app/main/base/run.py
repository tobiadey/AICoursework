'''
Intro
Authors Muhammad Masum Miah, Oluwatobi Adewunmi
'''

# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard



print("Running base.py")
print("Loading breast Fashion MNIST dataset")

# Import the dataset
df = pd.read_csv('./data/test.csv')
print(df.head())


# train_data = np.array(df,dtype='float32')
# #
# # print(train_data)
#
# # Pixel data (divided by 255 to rescale 0-1 not 0-255)
# X = train_data[:,1:]/255
# # First column (divided by 255 to rescale 0-1 not 0-255)
# y = train_data[:,0]

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# #
y = df['label']
X = df.drop('label',axis=1)
#
# image = X[16,:].reshape((28,28))
# plt.imshow(image)
# plt.show()





print("End of base.py")
print("\n")
