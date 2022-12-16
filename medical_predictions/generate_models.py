import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

hsptl_train_df = pd.read_parquet("../data/processed/hsptl_train.parquet")
hsptl_test_df = pd.read_parquet("../data/processed/hsptl_test.parquet")

X = hsptl_train_df.drop(['StayLength'], axis=1)
y = hsptl_train_df['StayLength']
rus = RandomOverSampler(random_state=42)
X, y = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier(max_depth=15)
bclf = BaggingClassifier(estimator=dtc, n_estimators=25, random_state=42)
bclf.fit(X_train,y_train)
y_pred = bclf.predict(hsptl_test_df)
pd.DataFrame(y_pred, columns=['pred']).to_csv('../data/predictions/akumoth.csv',index=False)