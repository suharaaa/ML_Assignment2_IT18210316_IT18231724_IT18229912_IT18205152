from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import CategoricalNB

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

dataset = pd.read_csv('data/bank-full.csv', header=0, delimiter=";")

feature_names = ["age", "job", "marital", "education", "balance", "housing", "loan"]

x = dataset[feature_names]
y = dataset["default"]

# Plot chart for frequency of default yes and no
print(y.value_counts())
sns.countplot(x='y', data=dataset)
plt.show()

# preprocess data
x.job = pd.Categorical(pd.factorize(x.job)[0])
x.marital = pd.Categorical(pd.factorize(x.marital)[0])
x.education = pd.Categorical(pd.factorize(x.education)[0])
x.housing = pd.Categorical(pd.factorize(x.housing)[0])
x.loan = pd.Categorical(pd.factorize(x.loan)[0])

x_train, x_test, y_train, y_test = train_test_split(x, y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

cnb = CategoricalNB()
cnb.fit(x_train, y_train)

y_pred = cnb.predict(x_test)

training_accuracy = cnb.score(x_train, y_train)
testing_accuracy = cnb.score(x_test, y_test)

print('Train Accuracy: {:.5f}'.format(training_accuracy))
print('Test Accuracy: {:.5f}'.format(testing_accuracy))

c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)

print(classification_report(y_test, y_pred))

