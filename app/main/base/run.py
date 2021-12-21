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

def processData(filename):
# Import the dataset
    df = pd.read_csv('./data/' + filename + '.csv')
    print("-------------------------print head of dataframe------------------------")
    print(df.head())
    print("-------------------------Check the number of classes------------------------")
    print(set(df['label']))
    print("-------------------------check minimum and maximum values in the feature variable columns------------------------")

    print([df.drop(labels='label', axis=1).min(axis=1).min(),
    df.drop(labels='label', axis=1).max(axis=1).max()])


#convert data from unsigned integers to float.
    train_data = np.array(df,dtype='float32')

# print(train_data)

# Features scaling
# Pixel data (divided by 255 to rescale to 0-1 and not 0-255)
    X = train_data[:,1:]/255
# As the target variable is the labels column(1st column), this will be our y variable.
    y = train_data[:,0]

#print the shape of X and y
    print('X: ' + str(X.shape))
    print('Y:  '+ str( y.shape))
    return X, y

# print('----------------------processing sample data--------------------------------')
# X, y = processData('sample')

print('----------------------processing training data--------------------------------')
X, y = processData('train')

# print('----------------------processing testing data--------------------------------')
# X_test, y_test = processData('test')


#function to the train the datasets, sample(used for testing purposes)  and actual
def trainData(X,y):
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.2, random_state = 1)
    return X_train,X_validate,y_train,y_validate


print('-------------------------------------using actual data--------------------------------')
X_train,X_validate,y_train,y_validate = trainData(X,y)
# print('-------------------------------------using sample data--------------------------------')
# X_train,X_validate,y_train,y_validate = trainData(X,y)



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



'''old'''

# '''
# Intro
# Authors Muhammad Masum Miah, Oluwatobi Adewunmi
# '''
#
# # import the necessary libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.metrics import accuracy_score
# # from sklearn.preprocessing import LabelEncoder
#
#
# print("Running base.py")
# print("Loading breast Fashion MNIST dataset")
#
# # Import the dataset
# df = pd.read_csv('./data/test.csv')
# print(df.head())
#
#
# train_data = np.array(df,dtype='float32')
#
# # print(train_data)
#
# # Pixel data (divided by 255 to rescale 0-1 not 0-255)
# X = train_data[:,1:]/255
# # First column (divided by 255 to rescale 0-1 not 0-255)
# y = train_data[:,0]
#
# print(y[:100])
# # print(y.shape[1])
# # # Splitting the dataset into the Training set and Test set
# # from sklearn.model_selection import train_test_split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#
#
# # image = X[500,:].reshape((28,28))
# # plt.imshow(image)
# # plt.show()
#
# # y = df['label']
# # X = df.drop('label',axis=1)
#
#
# print("End of base.py")
# print("\n")