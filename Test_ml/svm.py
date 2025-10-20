from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split #predicting categories
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, pred))
