# -*- coding: utf-8 -*-

"""
Created on Fri Dec 21 12:24:16 2018

@author: aditya
"""

# this program analyzes the importances of features

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn import model_selection

titanic_train= pd.read_csv("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\train.csv")
titanic_test= pd.read_csv("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\test.csv")

# Passenger's Name column consists of Titles - Mr. Ms. Jr. Sr etc.. which we could use to fill 
# missing Age values to fit them logically by calculating mean age of people grouped by their titles
# instead of just filling by mean of all Age values present
# But before doing this we can evaluate the feature importance for this title column

# concatenate train and test to get distinct values present across
# add a dummy Survived column in titanic_test to match the number of columns
titanic_test['Survived']= None

titanic= pd.concat([titanic_train, titanic_test])

def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

# the map function in python applies the passed function definition to each item in the 
# iterable object and returns a list containing all function call results
titanic['Title']= titanic['Name'].map(extract_title)

# fill missing data using pre processing imputer
mean_imputer= preprocessing.Imputer()
mean_imputer.fit(titanic[['Age','Fare']])
titanic.info()
# Age has missing values in train data. Fare has missing in both train and test
titanic[['Age','Fare']]= mean_imputer.transform(titanic[['Age','Fare']])

# Fill missing Embarked
titanic['Embarked']= np.where(titanic['Embarked'].isnull(),titanic['Embarked'].mode(),titanic['Embarked'])

titanic.info()


def convert_age_to_age_group(age):
    if(age>=0 and age<=12):
        return 'child'
    elif(age<20):
        return 'teenager'
    elif(age<30):
        return 'young-adult'
    elif(age<60):
        return 'middle-aged'
    elif(age>60):
        return 'old-aged'
    
#convert numerical age column to categorical age column
titanic['Age_group']= titanic['Age'].map(convert_age_to_age_group)

# create new column FamilySize combining SibSp, Parch and see if we could get any better pattern recognition
titanic['FamilySize']= titanic['SibSp']+ titanic['Parch'] + 1

def family_size_group(size):
    if(size==1):
        return 'Single'
    elif(size<=3):
        return 'Small'
    elif(size<6):
        return 'Mid-sized'
    else:
        return 'Large'
    
titanic['Family_size_group']= titanic['FamilySize'].map(family_size_group)

titanic.info()

# Now we have 3 new categorical columns: Title, Age_group, family_size_group
# one hot encode all the cat columns
titanic_1Hot= pd.get_dummies(titanic, columns=['Sex','Pclass','Embarked','Title','Age_group','Family_size_group'])
titanic_1Hot.info()

# drop unwanted columns in pattern recognition
titanic_refined=titanic_1Hot.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)

titanic_refined.shape

#Now split train and test data
x_train= titanic_refined[0:titanic_train.shape[0]]
x_train.shape
x_train.info()
y_train= titanic_train['Survived']

#now build the model
treeClassifier= tree.DecisionTreeClassifier()
dt_grid= {'max_depth':list(range(10,15)), 'min_samples_split':list(range(2,8)),'criterion':['gini','entropy']}

param_grid= model_selection.GridSearchCV(treeClassifier, dt_grid, cv=10)
param_grid.fit(x_train, y_train)

print(param_grid.best_score_)
print(param_grid.best_estimator_)
print(param_grid.best_params_)
print(param_grid.score(x_train,y_train)) # gets score on the full train data

# Let's see the feature importances computed by the decision tree
# create a dataframe with features and their importances

FI_df= pd.DataFrame({"feature": x_train.columns, "importance": param_grid.best_estimator_.feature_importances_}) 
print(FI_df)
# Title_Mr has more feature importance than any others. And most of them are of zero importance!

# now predict on the test data
x_test= titanic_refined[titanic_train.shape[0]:]
x_test.shape
x_test.info()
titanic_test['Survived']= param_grid.predict(x_test)

os.getcwd()
os.chdir('C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\')

titanic_test.to_csv('prediction_FI.csv', columns=['PassengerId','Survived'], index=False)

