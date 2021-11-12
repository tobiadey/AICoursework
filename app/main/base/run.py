# importing data sets and setting X and y

'''
Intro
Authors Muhammad Masum Miah, Oluwatobi Adewunmi
'''

# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


print("Running base.py")
print("Loading breast FootballData dataset")

# Import the dataset
df = pd.read_csv('./data/FootballData.csv')

# Encoding the Variables to replace strings with ints
columns = ["Teams", "Against", "Venue" ,"TeamsStadium", "Result"]
le = LabelEncoder()
df[columns] = df[columns].apply(le.fit_transform)

#Set X and y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# print(X)
# print(y)

# plt.scatter(X,y)
# plt.show();



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print("End of base.py")
print("\n")



def speak():
    print("hello world")
