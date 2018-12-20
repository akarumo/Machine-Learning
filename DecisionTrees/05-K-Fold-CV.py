# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:38:30 2018

@author: adkarumo
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import model_selection

titanic_train= pd.read_csv("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\train.csv")

titanic_train.info()

# filling missing values in Embraked
titanic_train['Embarked']= np.where(titanic_train['Embarked'].isnull(), titanic_train['Embarked'].mode(), titanic_train['Embarked'])
titanic_train.info()

# transform categorical columns to one hot encoding
titanic_train_1Hot= pd.get_dummies(titanic_train, columns=['Pclass','Sex','Embarked']) 
titanic_train_1Hot.info()

# drop unwanted columns (Age as well for now)
x_train= titanic_train_1Hot.drop(['PassengerId','Age','Cabin','Ticket','Name','Survived'], 1)
y_train= titanic_train['Survived']

dt= tree.DecisionTreeClassifier()
# build model using decision tree classifier
dt.fit(x_train, y_train)

# apply k-fold technique to find out Cross Validation Score
cv_scores= model_selection.cross_val_score(dt,x_train,y_train, cv=10)
# print list of scores computed on cross validation data
print(cv_scores)
# mean of the cv scores computed
print(cv_scores.mean())

# compute score without Cross Validation. In other words, on the full train data
dt.score(x_train, y_train)


# use max_depth, min_sample_split in building decision tree
dt2= tree.DecisionTreeClassifier(max_depth=6, min_samples_split=2)
dt2.fit(x_train, y_train)
cv_scores2= model_selection.cross_val_score(dt2, x_train, y_train, cv=10)
print(cv_scores2)
print(cv_scores2.mean())


# Use grid search method to pass multiple parameters in building decision tree
dt3= tree.DecisionTreeClassifier()
param_grid= {'max_depth':[5,8,10], 'min_samples_split':[2,4,5], 'criterion':['gini','entropy']}
dt3_grid= model_selection.GridSearchCV(dt3, param_grid, cv=10, n_jobs=5)
dt3_grid.fit(x_train, y_train)
dt3_grid.cv_results_
print(dt3_grid.best_score_)
# compute .score on full train data
print(dt3_grid.score(x_train, y_train))










