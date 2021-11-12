
from app.main.base.run import X,y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print("inside lecture 1 content")
print(X_train)
print(X_test)
print(y_train)
print(y_test)
