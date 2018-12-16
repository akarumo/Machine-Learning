# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 10:47:49 2018

@author: aditya
"""

import pandas as pd
import numpy as np
import os
from sklearn import tree

# read the train data
titanic_train= pd.read_csv("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\train.csv")

titanic_train.info()

# Embarked has only two values missing that can be filled with mode
titanic_train['Embarked']= np.where(titanic_train['Embarked'].isnull(), titanic_train['Embarked'].mode(), titanic_train['Embarked'])
titanic_train.info()

# transform non-numeric columns to One hot encoded columns
# PClass is a numeric column but it is a categorical column having class1, class2, class3 values logically

titanic_train_1_Hot= pd.get_dummies(titanic_train, columns=['Pclass','Sex','Embarked']) 

titanic_train_1_Hot.info()


# drop unwanted columns (drop Age for now)
train_x= titanic_train_1_Hot.drop(['PassengerId','Age','Cabin','Ticket','Name','Survived'], 1)
train_y= titanic_train['Survived']

# use entropy algorithm in building the Decision Tree
dt= tree.DecisionTreeClassifier(criterion= 'entropy')

# fit method builds the model using entropy based decision tree algorithm
dt.fit(train_x,train_y)

# Now predict the Survived column values in test data using the model built

# read the test data
titanic_test= pd.read_csv("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\test.csv")
titanic_test.info()
# Fare column has one value missing, it can be fiiled with the mean
titanic_test['Fare']= np.where(titanic_test['Fare'].isnull(), titanic_test['Fare'].mean(), titanic_test['Fare'])

# one hot encode the categorical columns
titanic_test_1Hot= pd.get_dummies(titanic_test, columns=['Pclass','Sex','Embarked'])

# drop unwanted columns just the same way we did for train data
test_x= titanic_test_1Hot.drop(['PassengerId','Age','Cabin','Ticket','Name'], axis=1)
titanic_test['Survived']=dt.predict(test_x)

os.getcwd()
os.chdir("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic")
titanic_test.to_csv("prediction1.csv", columns=['PassengerId','Survived'], index=False)

