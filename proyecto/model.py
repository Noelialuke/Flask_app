import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("data.csv")

X = df[["Altura", "Peso"]]
y = df["Sexo"]

clf = GaussianNB() 
clf.fit(X, y)

import joblib

joblib.dump(clf, "clf.pkl")
