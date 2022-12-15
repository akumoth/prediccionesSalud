import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix 

hsptl_train_df = pd.read_parquet("../data/processed/hsptl_train.parquet")
X = hsptl_train_df.drop(['StayLength'], axis=1)
y = hsptl_train_df['StayLength']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42).fit(X_train, y_train)
print(logreg.score(X_test,y_test))

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
pred = dtc.predict(X_test)
print(classification_report(y_test, pred))