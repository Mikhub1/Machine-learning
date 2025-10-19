import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

conn = sqlite3.connect('flowers.db')

df = pd.read_sql_query("SELECT * FROM iris_data", conn)

X = df.drop('species', axis=1)  
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred))


conn.close()
