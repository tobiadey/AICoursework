'''
Intro
Authors Muhammad Masum Miah, Oluwatobi Adewunmi
'''

# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

print("Running base.py")
print("Loading breast Fashion MNIST dataset")

def processData():
    # Import the dataset
    df_train = pd.read_csv('./data/' + 'fashionTrain.csv')
    df_test = pd.read_csv('./data/' + 'fashionTest.csv')

    print("-------------------------print head of dataframe (train)------------------------")
    print(df_train.head())
    print("-------------------------print head of dataframe (test)------------------------")
    print(df_test.head())


    print("-------------------------Check the number of classes(train)------------------------")
    print(set(df_train['label']))
    print("-------------------------Check the number of classes(test)------------------------")
    print(set(df_test['label']))

    print("-------------------------check minimum and maximum values in the feature variable columns------------------------")

    print("-------------------------Checking min & max values for (train data)------------------------")
    print([df_train.drop(labels='label', axis=1).min(axis=1).min(),
    df_train.drop(labels='label', axis=1).max(axis=1).max()])

    print("-------------------------Checking min & max values for (test data)------------------------")
    print([df_test.drop(labels='label', axis=1).min(axis=1).min(),
    df_test.drop(labels='label', axis=1).max(axis=1).max()])


#convert data from unsigned integers to float.
    train_data = np.array(df_train,dtype='float32')
    test_data = np.array(df_test,dtype='float32')

    # print(train_data)

    # Features scaling
    # Pixel data (divided by 255 to rescale to 0-1 and not 0-255)
    # As the target variable is the labels column(1st column), this will be our y variable.

    # training data
    x_train = train_data[:,1:]/255
    y_train = train_data[:,0]

    # testing data
    x_test = test_data[:,1:]/255
    y_test = test_data[:,0]

    #print the shape of X and y
    print('X_train: ' + str(x_train.shape))
    print('Y_train:  '+ str( y_train.shape))
    print('X_test: ' + str(x_test.shape))
    print('Y_test:  '+ str( y_test.shape))


    return x_train, y_train,x_test,y_test





# print('----------------------processing sample data--------------------------------')
# X, y = processData('sample')

print('----------------------processing training data--------------------------------')
X, y,X_test,y_test = processData()


#function to the train the datasets, sample(used for testing purposes)  and actual
def trainData(X,y):
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.2, random_state = 1)
    return X_train,X_validate,y_train,y_validate

X_train,X_validate,y_train,y_validate = trainData(X,y)




# plot images to show what our dataset looks like.
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X[:25][i].reshape(28,28), cmap='gray')
plt.tight_layout()
plt.show()


print("End of base.py")
print("\n")
