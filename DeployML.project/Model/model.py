import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("./Model/glass.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:,9]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


model = RandomForestClassifier() #criterion='entropy', n_estimators=300, random_state=123
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('ACCURACY is ', accuracy_score(y_test, y_pred)*100, '%')

# yyy = model.predict([])

# save the model to disk
filename = './Model/ML_model.joblib'
dump(model, filename)

# lis = ['1', '1', '2', '3', '1', '3', '1', '1', '1']
# model_loaded = joblib.load('./Model/ML_model.joblib')
# y_pred_loaded = model_loaded.predict([lis])