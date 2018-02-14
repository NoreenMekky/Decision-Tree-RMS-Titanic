# -*- coding: utf-8 -*-
#from __future__ import print_function
#Author: Noreen Mekky

from sklearn import preprocessing
from sklearn import tree
import pandas as pd
import os

os.getcwd()

#getting train and test csv files 
train = pd.read_csv("train.csv", header = 0)
test = pd.read_csv("test.csv", header = 0)
#handling missing values of age and fare by replacing it with the median
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)


# Convert sex and embarked variables to numeric
label_encoder = preprocessing.LabelEncoder()


train['Sex']=label_encoder.fit_transform(train['Sex'].astype('str'))
train['Embarked']=label_encoder.fit_transform(train['Embarked'].astype('str'))

test['Sex']=label_encoder.fit_transform(test['Sex'].astype('str'))
test['Embarked']=label_encoder.fit_transform(test['Embarked'].astype('str'))

train.info()

##############################

featureX= train[["Pclass", "Sex", "Age", "SibSp", "Parch","Fare", "Embarked"]]
featureY= train[["Survived"]]

#classification
clf = tree.DecisionTreeClassifier()
clf = clf.fit(featureX,featureY)
prediction = clf.predict(test[["Pclass", "Sex", "Age", "SibSp", "Parch","Fare", "Embarked"]])


# Generate prediction File containing 
# 2 coloumns for passenger id and survival status
# 1 for survived and zero for not survived
StackingSubmission = pd.DataFrame({ 'PassengerId': test["PassengerId"],
                           'Survived': prediction })
# NOTE: you should specify the path for 
# which you want to save the result file
# if you are running it on your machine
StackingSubmission.to_csv("/home/noreen/Desktop/results.csv", index=False)
