#Authors Muhammad Masum Miah, Oluwatobi Adewunmi
# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import the dataset
df = pd.read_csv('./FootballData.csv')


# Encoding the Variables to replace strings with ints
columns = ["Teams", "Against", "Venue" ,"TeamsStadium", "Result"]
le = LabelEncoder()
df[columns] = df[columns].apply(le.fit_transform)

#Set X and y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# print(X)
# print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)




'''
above this should not bee neded as we should be able to import base.py ther
'''

'''
Well start adding these to the lecture folders that they are expected to be in
'''
#Decision tree
decision_tree = DecisionTreeClassifier(criterion = 'entropy')
decision_tree.fit(X_train,y_train)
y_pred = decision_tree.predict(X_test)
# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#k neighbours
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
kn.fit(X_train, y_train)
y_pred = kn.predict(X_test)
# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# # Training the Logistic Regression model on the Training set
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(random_state = 0)
# lr.fit(X_train, y_train)
# y_pred = lr.predict(X_test)
# # Evaluate accuracy
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Training the SVM model on the Training set
from sklearn.svm import SVC
svm1 = SVC(kernel = 'linear', random_state = 0)
svm1.fit(X_train, y_train)
y_pred = svm1.predict(X_test)
# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
