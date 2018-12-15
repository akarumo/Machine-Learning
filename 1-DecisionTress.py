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

# consider non-categorical and non-missing data columns

