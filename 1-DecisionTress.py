# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:39:00 2018

@author: aditya
"""

import pandas as pd
import os
from sklearn import tree # imports just tree from sklearn library

# read titanic train data
titanic_train= pd.read_csv("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\train.csv")

# check the data type of the loaded csv using pandas read csv
print(type(titanic_train))

# check the structure and details of the data frame

print(titanic_train.shape)
print(titanic_train.info())
print(titanic_train.describe())

# consider non-categorical and non-missing data columns on X-axis and Survived on Y-axis
x_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
y_titanic_train = titanic_train['Survived'] #Y-Axis

#Build the decision tree model using defaults. The default algorithm criterion used is Gini 
dt = tree.DecisionTreeClassifier()
dt.fit(x_titanic_train, y_titanic_train)

# Now predict the outcome using decision tree model built above
#Read the Test Data
titanic_test = pd.read_csv("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\test.csv")
x_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#Use .predict method on Test data using the model which we built
titanic_test['Survived'] = dt.predict(x_test) 
os.getcwd() #To get current working directory
os.chdir("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\")
titanic_test.to_csv("prediction0.csv", columns=['PassengerId','Survived'], index=False)