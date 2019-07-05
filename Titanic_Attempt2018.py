# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 22:25:13 2018

@author: kilia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import accuracy_score

data_train = pd.read_csv("C:/Users/kilia/OneDrive/Documents/Kaggle/Titanic/train.csv")
X_train = data_train.drop('Survived', axis=1)
y_train = data_train['Survived']
data_train.head()
len(data_train)

data_test = pd.read_csv("C:/Users/kilia/OneDrive/Documents/Kaggle/Titanic/test.csv")
X_test = data_test

data_trainSurv = data_train[data_train['Survived']==1]
data_trainPer = data_train[data_train['Survived']==0]

plt.figure()
sns.swarmplot(x='Survived', y= data_train['Age'], data_train= data_train, palette = 'Set1');
plt.show()

# based on above, age does not seem like a good predictor

sexb = data_train['Sex'].map({'male':1, 'female':0})
#data_train[data_train['Survived']==1 & sexb==1]
len(data_train[np.logical_and(data_train['Survived']==1,sexb==1)]) # males survived = 109
len(data_train[np.logical_and(data_train['Survived']==0,sexb==1)]) # males perished = 468
len(data_train[np.logical_and(data_train['Survived']==1,sexb==0)]) # females survived = 233
len(data_train[np.logical_and(data_train['Survived']==0,sexb==0)]) # females perished = 81

ms = (data_train[np.logical_and(data_train['Survived']==1,sexb==1)])
mp = (data_train[np.logical_and(data_train['Survived']==0,sexb==1)]) 
fs = (data_train[np.logical_and(data_train['Survived']==1,sexb==0)]) 
fp = (data_train[np.logical_and(data_train['Survived']==0,sexb==0)])
# not sure how to visulaise this

# based on above, sex should be important!!

plt.figure()
sns.swarmplot(x='Survived', y= data_train['SibSp'], data_train= data_train, palette = 'Set1');
plt.show()

# might be useful

plt.figure()
sns.swarmplot(x='Survived', y= data_train['Parch'], data_train= data_train, palette = 'Set1');
plt.show()

# doesn't seem all that helpful, very little much of a sample beyond Parch = 2

plt.figure()
sns.swarmplot(x='Survived', y= data_train['Fare'], data_train= data_train, palette = 'Set1');
plt.show()

# might be helpful

plt.figure()
sns.swarmplot(x='Survived', y= data_train['Pclass'], data_train= data_train, palette = 'Set1');
plt.show()


Pclass = data_train['Pclass']
one_s = (data_train[np.logical_and(data_train['Survived']==1,Pclass==1)])
one_p = (data_train[np.logical_and(data_train['Survived']==0,Pclass==1)])
two_s = (data_train[np.logical_and(data_train['Survived']==1,Pclass==2)])
two_p = (data_train[np.logical_and(data_train['Survived']==0,Pclass==2)])
three_s = (data_train[np.logical_and(data_train['Survived']==1,Pclass==3)])
three_p = (data_train[np.logical_and(data_train['Survived']==0,Pclass==3)])

xvol = []
xvol = [len(one_s),len(one_p),len(two_s),len(two_p),len(three_s),len(three_p)]
s = [xvol[i] for i in range(len(xvol))]
plt.scatter(Pclass, data_train['Survived'], s=s) # trying to create bubble plot here

# =============================================================================
# Fitting Algos
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScalar, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer

X_train = X_train.drop('Name', axis=1)
X_train = X_train.drop('Ticket', axis=1)
X_train = X_train.drop('Cabin', axis=1) # so many Nan's ##Edit(Jerome) don't think we should drop these just yet, from looking at the data in the training set every passanger with a cabin listed survived. so maybe they only found out what cabin they were in from the survivors after the ship sank.
X_train = X_train.drop('PassengerId', axis=1)

labelencoder_X = LabelEncoder()
X_train['Pclass'] = labelencoder_X.fit_transform(X_train['Pclass'])
X_train['Sex'] = labelencoder_X.fit_transform(X_train['Sex'])

len(X_train[X_train['Embarked']=='S'])
len(X_train[X_train['Embarked']=='C'])
len(X_train[X_train['Embarked']=='Q'])
len(X_train[X_train['Embarked']==' nan']) # no way of dealing with nan's this way

X_train['Embarked'].fillna('S', inplace=True) # used S as it is mode
X_train['Embarked'] = labelencoder_X.fit_transform(X_train['Embarked'])

np.mean(X_train['Age'])
X_train['Age'].fillna(np.mean(X_train['Age']), inplace=True)

X_Pclass = pd.get_dummies(X_train['Pclass'], prefix=['Pclass'], drop_first=True) # onehotencoding dataframe
X_Embarked = pd.get_dummies(X_train['Embarked'], prefix=['Embarked'], drop_first=True)

X_train = X_train.drop('Pclass', axis=1)
X_train = X_train.drop('Embarked', axis=1)
X_train = X_train.append(X_Pclass)
X_train = X_train.append(X_Embarked)

sc = StandardScalar()
X_train = sc.fit_transform(X_train)
X_train = sc.transform(X_train)

knn = KNeighborsClassifier(n_neighbors = 5, metric='minkowski',p=2)
knn.fit(X_train, y_train)

