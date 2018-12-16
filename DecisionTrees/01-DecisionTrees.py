# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:27:27 2018

@author: adkarumo
"""

import pandas as pd # imports pandas library
from sklearn import tree # imports tree from sklearn
import os

# Titanic: Machine Learning from Disaster
# Kaggle link: https://www.kaggle.com/c/titanic/data

# Read titanic train data using pandas library function read_csv
titanic_train= pd.read_csv("D:\\Users\\adkarumo\\Documents\\Data Science\\Titanic\\train.csv")

# Check the data type of the loaded csv file
print(type(titanic_train))

# Check the metadata details of the data frame, running these will print respective outputs
titanic_train.shape
titanic_train.info()
titanic_train.describe()

# For simplicity, choose non-null and significant columns in predicting Survived
x_train = titanic_train[['Pclass', 'SibSp', 'Parch']] # X-Axis
y_train= titanic_train['Survived'] # Y-Axis

# Select Decision Tree Classifier as we will need to classify the test data into Survived/Non-Survived Passengers
dt= tree.DecisionTreeClassifier()

# Use train data to build the Decision Tree Model
model= dt.fit(x_train,y_train)

# Now we have the model ready, load the test data from the test.csv file
titanic_test= pd.read_csv("D:\\Users\\adkarumo\\Documents\\Data Science\\Titanic\\test.csv")

# Choose the same columns from test as that of train data
x_test= titanic_test[['Pclass', 'SibSp', 'Parch']]
# Run predict on the test data using the model we build earlier
titanic_test['Survived']= model.predict(x_test)

# Verify the current working directory
os.getcwd()
# Change the directory if needed to write the output into desired destination
os.chdir("D:\\Users\\adkarumo\\Documents\\Data Science\\Titanic")

# Write the predicted values to the csv
titanic_test.to_csv("titanic_prediction_1.csv", columns=['PassengerId','Survived'], index=False)
