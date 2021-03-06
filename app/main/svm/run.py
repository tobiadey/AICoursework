'''
Using LinearRegression & SVM models on fashion data
'''
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC # "Support vector classifier"
from sklearn.metrics import accuracy_score,confusion_matrix
from app.main.base.run import X_train,X_validate,y_train,y_validate,X_test,y_test


# Regression chart.
def chart_regression(name,pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    # plt.title(name)
    plt.title('Comparison of Actual value to Predicted value for ' + name + ' data')

    plt.show();


#function to create models
# 'lr' for linear regression,
# 'svm' for svm
    # which is followed by either 'default' or a specific value for 'C'

def createModel(modelType,*c):
    if modelType== 'lr':
        model = LinearRegression()

    elif modelType == 'svm':
        if c[0] == 'default':
            model = SVC()
        else:
            model = SVC(gamma='scale', kernel='rbf', C=c[0])


    #Train the model using the training sets
    model.fit(X_train,y_train)

    #Predict the response for validation dataset
    y_pred = model.predict(X_validate)





    print(" -----------------------------use this trained model to predict the value for the validation data--------------------------")
    print(" -----------------------------compare prediction to actual values ----------------------------------")

    # #build a new data frame with two columns, the actual values of the validation data,
    # #and the predictions of the model
    # for validation data
    df_compare = pd.DataFrame({'Actual': y_validate, 'Predicted': y_pred})
    df_head = df_compare.head(25)
    print(df_head)

    if modelType== 'lr':


        y_pred2 = model.predict(X_test)

        # for test data
        df_compare2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
        df_head2 = df_compare2.head(25)
        print(df_head2)

        # print("\n=================================================================================================================\n")
        # print("Testing someeeeeeee----------------------------------")
        # print("coeff"+ str(model.coef_))
        # print("intercept "+ str(model.intercept_) )




    print(" -----------------------------visual representation and mean & rms calculated----------------------------------")
    print(" -----------------------------less than 10% of the means is often quoted as being a reasonably good score.----------------------------------")

    # for validation data
    print('\nMean:', np.mean(y_validate))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_validate, y_pred)))

    if modelType == 'svm':

        # save model


        # for validation data
        # Model Accuracy: how often is the classifier correct?
        print("\n=================================================================================================================\n")
        print('\nthe accuracy of SVM with c value ' + str(c[0]) +' is : ' + str(accuracy_score(y_validate, y_pred, normalize=True)))
        cm = confusion_matrix(y_validate, y_pred)
        print('\nconfusion_matrix:', "\n", cm)


        '''uncomment to run test on test fashionTest data (Note : takes longer to run)'''
        # for test data
        # Model Accuracy: how often is the classifier correct?
        # print("\n=================================================================================================================\n")
        # print('\nthe accuracy of SVM with c value ' + str(c[0]) +' is : ' + str(accuracy_score(y_test, y_pred2, normalize=True)))
        # cm2 = confusion_matrix(y_test, y_pred2)
        # print('\nconfusion_matrix:', "\n", cm2)

    elif modelType == 'lr':
        # for test data
        print('\ntest data')
        print('\nMean:', np.mean(y_test))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))

        print(" -----------------------------model coef ----------------------------------")
        print(model.coef_)
        print(model.intercept_)

        # plot the predictions versus actual results (with the x=y line down the middle
        # Regression chart
        chart_regression('Validation',y_pred[:100].flatten(), y_validate[:100], sort=True)
        df_head.plot(kind='bar', figsize=(10, 8))
        plt.title('comparison of actual value to Predicted value for validation data')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show();

        chart_regression('Testing',y_pred2[:100].flatten(), y_test[:100], sort=True)
        df_head.plot(kind='bar', figsize=(10, 8))
        plt.title('comparison of actual value to Predicted value for testing data')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show();



createModel('lr')

print("\n -----------------------------repeat above using a support vector machine for regression sklearn.sv.SVR--------------------\n")

'''C value 6 gives the best accuracy'''
#create multiple svm models while trialling different C values


print("SVM with default parameters...loading")
createModel('svm', 'default')
#the accuracy of SVM with default params is : 0.8625

'''createModel('svm', 4)'''
#the accuracy of SVM with c value 4 is : 0.877

print("\nSVM with c value 6...loading")
createModel('svm', 6)
#the accuracy of SVM with c value 6 is : 0.8815

# createModel('svm', 8)
#the accuracy of SVM with c value 8 is : 0.881

'''createModel('svm', 12)'''
#the accuracy of SVM with c value 12 is : 0.8805




'''old code'''


# #testing code
# #
# # X_train = X_train.to_numpy()
# # y_train = y_train.to_numpy()
# # X_test = X_test.to_numpy()
# # y_test = y_test.to_numpy()
# #
# #
# # for i in range(25):
# #     plt.subplot(5,5,i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.imshow(X_train[i].reshape(28,28)) #imshow  takes an array ( with dimension = 2, RGB or B/W) and gives you the image that corresponds to it
# #     plt.show()
#
#
# from app.main.base.run import X,y
#
# # y = tensorflow.keras.utils.to_categorical(y)
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#
#
# # Regression chart.
# def chart_regression(pred, y, sort=True):
#     t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
#     if sort:
#         t.sort_values(by=['y'], inplace=True)
#     plt.plot(t['y'].tolist(), label='expected')
#     plt.plot(t['pred'].tolist(), label='prediction')
#     plt.ylabel('output')
#     plt.legend()
#     plt.show();
#
# '''method 1'''
# # # play around with values to see if any improvement
#
# def createModel(modelType,*c):
#     if modelType== 'lr':
#         model = LinearRegression()
#
#
#     elif modelType == 'svm':
#         if c[0] == 'default':
#             model = SVC()
#         else:
#             model = SVC(gamma='scale', kernel='rbf', C=c[0])
#     # #Train the model using the training sets
#     model.fit(X_train,y_train)
#     # #Predict the response for test dataset
#     y_pred = model.predict(X_test)
#     # Model Accuracy: how often is the classifier correct?
#     print("\n=================================================================================================================\n")
#
#     print(
#         " -----------------------------use this trained model to predict the value for the testing data--------------------------")
#     print(" -----------------------------compare prediction to actual values ----------------------------------")
#
#     # #build a new data frame with two columns, the actual values of the test data,
#     # #and the predictions of the model
#     df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#     df_head = df_compare.head(25)
#     print(df_head)
#
#     print(" -----------------------------visual rep and mean & rms calculated----------------------------------")
#     print(
#         " -----------------------------less than 10% of the means is often quoted as being a reasonably good score.----------------------------------")
#     print('\nMean:', np.mean(y_test))
#     print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
#
#     if modelType == 'svm':
#         print('\nthe accuracy of SVM with c value ' + str(c[0]) +' is : ' + str(accuracy_score(y_test, y_pred, normalize=True)))
#         cm = confusion_matrix(y_test, y_pred)
#         print('\nconfusion_matrix:', "\n", cm)
#     elif modelType == 'lr':
#         print(" -----------------------------model coef ----------------------------------")
#         print(model.coef_)
#         # plot the predictions versus actual results (with the x=y line down the middle
#         # Regression chart
#         chart_regression(y_pred[:100].flatten(), y_test[:100], sort=True)
#         df_head.plot(kind='bar', figsize=(10, 8))
#         plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
#         plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#         plt.show();
#
#
# createModel('lr')
#
# print("\n -----------------------------repeat above using a support vector machine for regression sklearn.sv.SVR--------------------\n")
#
# createModel('svm', 'default')
#
# #
#
# ''' method 2'''
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# print(model.coef_)
#
# print(" -----------------------------use this trained model to predict the value for the testing data--------------------------")
# print(" -----------------------------compare prediction to actual values ----------------------------------")
#
# #calculate the predictions of the linear regression model
# y_pred = model.predict(X_test)
#
# #build a new data frame with two columns, the actual values of the test data,
# #and the predictions of the model
# df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# df_head = df_compare.head(25)
# print(df_head)
#
# print(" -----------------------------visual rep and mean & rms calculated----------------------------------")
# print(" -----------------------------less than 10% of the means is often quoted as being a reasonably good score.----------------------------------")
#
# df_head.plot(kind='bar',figsize=(10,8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show();
#
#
# from sklearn import metrics
#
# print('Mean:', np.mean(y_test))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
# print(" -----------------------plot the predictions versus actual results (with the x=y line down the middle----------------------------------")
# print(" -----------------------------Regression chart----------------------------------")
#
#
# # Regression chart.
# def chart_regression(pred, y, sort=True):
#     t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
#     if sort:
#         t.sort_values(by=['y'], inplace=True)
#     plt.plot(t['y'].tolist(), label='expected')
#     plt.plot(t['pred'].tolist(), label='prediction')
#     plt.ylabel('output')
#     plt.legend()
#     plt.show();
#
# chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)
#
#
# print(" -----------------------------repeat above using a support vector machine for regression sklearn.sv.SVR--------------------")
#
# import tensorflow.keras.utils
#
#
# from sklearn.svm import SVC # "Support vector classifier"
# from sklearn.metrics import accuracy_score,confusion_matrix
#
#
#
#
#
# def createSVM(c):
#     # play around with values to see if any improvement
#     if c == 'default':
#         svm_model = SVC()
#     else:
#         svm_model = SVC(gamma='scale', kernel='rbf', C=c)
#
#     print(X_train.shape)
#     print(y_train.size)
#     # Train the model using the training sets
#     svm_model.fit(X_train, y_train)
#
#     # Predict the response for test dataset
#     y_pred2 = svm_model.predict(X_test)
#
#     # build a new data frame with two columns, the actual values of the test data,
#     # and the predictions of the model
#     df_compares = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
#     df_heads = df_compares.head(25)
#     print(df_heads)
#
#     # Model Accuracy: how often is the classifier correct?
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred2))
#     print('Mean:', np.mean(y_test))
#     print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
#     cm = confusion_matrix(y_test, y_pred2)
#     print('confusion_matrix:', "\n", cm)
#
#
# createSVM(5)

'''method 3'''

# # build the model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# print(model.coef_)
#
# print(" -----------------------------use this trained model to predict the value for the testing data--------------------------")
# print(" -----------------------------compare prediction to actual values ----------------------------------")
#
# #calculate the predictions of the linear regression model
# y_pred = model.predict(X_validate)
# y_pred2 = model.predict(X_test)
#
# #build a new data frame with two columns, the actual values of the test data,
# #and the predictions of the model
# df_compare = pd.DataFrame({'Actual': y_validate, 'Predicted': y_pred})
# df_head = df_compare.head(25)
# print(df_head)
#
# df_compare2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
# df_head2 = df_compare2.head(25)
# print(df_head2)
#
#
# print(" -----------------------------visual rep and mean & rms calculated----------------------------------")
# print(" -----------------------------less than 10% of the means is often quoted as being a reasonably good score.----------------------------------")
#
# df_head.plot(kind='bar',figsize=(10,8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show();
#
#
# from sklearn import metrics
#
# print('Mean:', np.mean(y_test))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
# print(" -----------------------plot the predictions versus actual results (with the x=y line down the middle----------------------------------")
# print(" -----------------------------Regression chart----------------------------------")
#
#
# # Regression chart.
# def chart_regression(pred, y, sort=True):
#     t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
#     if sort:
#         t.sort_values(by=['y'], inplace=True)
#     plt.plot(t['y'].tolist(), label='expected')
#     plt.plot(t['pred'].tolist(), label='prediction')
#     plt.ylabel('output')
#     plt.legend()
#     plt.show();
#
# chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)
#

# print(" -----------------------------repeat above using a support vector machine for regression sklearn.sv.SVR--------------------")
#
# import tensorflow.keras.utils
#
#
# from sklearn.svm import SVC # "Support vector classifier"
# from sklearn.metrics import accuracy_score,confusion_matrix
#
# # play around with values to see if any improvement
# svm_model = SVC()
#
# print(X_train.shape)
# print(y_train.size)
# #Train the model using the training sets
# svm_model.fit(X_train, y_train)
#
# #Predict the response for test dataset
# y_pred2 = svm_model.predict(X_validate)
# y_pred22 = svm_model.predict(X_test)
#
# #build a new data frame with two columns, the actual values of the test data,
# #and the predictions of the model
# df_compares = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
# df_heads = df_compares.head(25)
# print(df_heads)
#
# df_compares2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred22})
# df_heads2 = df_compares2.head(25)
# print(df_heads2)
#
# # Model Accuracy: how often is the classifier correct?
# print('Accuracy: %.2f' % accuracy_score(y_validate, y_pred2))
# print('Mean:', np.mean(y_test))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_validate, y_pred2)))
# cm = confusion_matrix(y_validate, y_pred2)
# print('confusion_matrix:',"\n",cm)
#
# # Model Accuracy: how often is the classifier correct?
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred22))
# print('Mean:', np.mean(y_test))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred22)))
# cm = confusion_matrix(y_test, y_pred22)
# print('confusion_matrix:',"\n",cm)
