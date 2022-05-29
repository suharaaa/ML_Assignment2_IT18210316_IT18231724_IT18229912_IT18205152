from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

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
sns.countplot(x='y', data=dataset)
plt.show()

# preprocess data
x.loc[:, 'job'] = pd.Categorical(pd.factorize(x.job)[0])
x.loc[:, 'marital'] = pd.Categorical(pd.factorize(x.marital)[0])
x.loc[:, 'education'] = pd.Categorical(pd.factorize(x.education)[0])
x.loc[:, 'housing'] = pd.Categorical(pd.factorize(x.housing)[0])
x.loc[:, 'loan'] = pd.Categorical(pd.factorize(x.loan)[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lreg = LogisticRegression()
lreg.fit(x_train, y_train)

y_pred = lreg.predict(x_test)

training_accuracy = lreg.score(x_train, y_train)
testing_accuracy = lreg.score(x_test, y_test)

print('Train Accuracy: {:.5f}'.format(training_accuracy))
print('Test Accuracy: {:.5f}'.format(testing_accuracy))

c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)

print(classification_report(y_test, y_pred))

