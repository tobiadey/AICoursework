'''
Using LinearRegression & SVM models on fashion data
'''
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#testing code
#
# X_train = X_train.to_numpy()
# y_train = y_train.to_numpy()
# X_test = X_test.to_numpy()
# y_test = y_test.to_numpy()
#
#
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(X_train[i].reshape(28,28)) #imshow  takes an array ( with dimension = 2, RGB or B/W) and gives you the image that corresponds to it
#     plt.show()


from app.main.base.run import X,y

# y = tensorflow.keras.utils.to_categorical(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# build the model
model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)

print(" -----------------------------use this trained model to predict the value for the testing data--------------------------")
print(" -----------------------------compare prediction to actual values ----------------------------------")

#calculate the predictions of the linear regression model
y_pred = model.predict(X_test)

#build a new data frame with two columns, the actual values of the test data,
#and the predictions of the model
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

print(" -----------------------------visual rep and mean & rms calculated----------------------------------")
print(" -----------------------------less than 10% of the means is often quoted as being a reasonably good score.----------------------------------")

df_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show();


from sklearn import metrics

print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(" -----------------------plot the predictions versus actual results (with the x=y line down the middle----------------------------------")
print(" -----------------------------Regression chart----------------------------------")


# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show();

chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)


print(" -----------------------------repeat above using a support vector machine for regression sklearn.sv.SVR--------------------")

import tensorflow.keras.utils


from sklearn.svm import SVC # "Support vector classifier"
from sklearn.metrics import accuracy_score

# play around with values to see if any improvement
svm_model = SVC()

print(X_train.shape)
print(y_train.size)
#Train the model using the training sets
svm_model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred2 = svm_model.predict(X_test)

#build a new data frame with two columns, the actual values of the test data,
#and the predictions of the model
df_compares = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
df_heads = df_compares.head(25)
print(df_heads)

# Model Accuracy: how often is the classifier correct?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred2))
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))







# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#
# from sklearn.svm import SVC
# classifier =SVC(gamma='scale',kernel='rbf',C=8)
# classifier.fit(X_train, y_train)

#
# y_pred = classifier.predict(X_test)
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print('the accuracy of SVM is : ' + str(accuracy_score(y_test, y_pred, normalize=True)))
