from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd

dataset = pd.read_csv('data/bank-full.csv', header=0, delimiter=";")

feature_names = ["age", "job", "marital", "education", "balance", "housing", "loan"]

x = dataset[feature_names]
y = dataset["default"]

# preprocess data
x.job = pd.Categorical(pd.factorize(x.job)[0])
x.marital = pd.Categorical(pd.factorize(x.marital)[0])
x.education = pd.Categorical(pd.factorize(x.education)[0])
x.housing = pd.Categorical(pd.factorize(x.housing)[0])
x.loan = pd.Categorical(pd.factorize(x.loan)[0])


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Accuracy: {:.2f}'.format(logreg.score(X_test, y_test)))
