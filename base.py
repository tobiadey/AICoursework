# importing data sets and setting X and y


'''
Intro
'''
print("Running base.py")
print("Loading breast cancer dataset")
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
iris = datasets.load_breast_cancer()
X = iris.data
y = iris.target
variety = iris.target_names


# We create a learnset from the sets above. We use permutation from np.random to split the data randomly.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)


print("End of base.py")
print("\n")

print("data:", iris.data)
print("\n")
print("target:", iris.target)
print("\n")
print("variety:", iris.target_names)

def speak():
    print("hello world")
