import numpy as np
import pandas as pd
import joblib

dataset = pd.read_csv("./Model/glass.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:,9]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(criterion='entropy', n_estimators=300, random_state=42)
cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)

print('ACCURACY is ', cls.score(X_test, y_test)*100, '%')

# yyy = cls.predict([])

# save the model to disk
filename = './finalized_model.sav'
joblib.dump(cls, filename)

# lis = ['1', '1', '2', '3', '1', '3', '1', '1', '1']
# cls_loaded = joblib.load('./finalized_model.sav')
# y_pred_loaded = cls_loaded.predict([lis])