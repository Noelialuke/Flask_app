import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

df = pd.read_csv("data.csv")

X = df[["Altura", "Peso"]]
y = df["Sexo"]

clf = GaussianNB() 
clf.fit(X, y)
joblib.dump(clf, "clf.pkl")
