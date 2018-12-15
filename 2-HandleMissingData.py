# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:10:10 2018

@author: aditya
"""

import pandas as pd
import numpy as np

titanic_train= pd.read_csv("C:\\Users\\aditya\\Documents\\0 Data Science\\Titanic\\train.csv")

titanic_train.info()

# =============================================================================
# titanic_train['Embarked'][titanic_train['Embarked'].isnull()] = titanic_train['Embarked'].mode()
# titanic_train.replace()
# =============================================================================

# Embarked has only two values missing that can be filled with mode
titanic_train['Embarked']= np.where(titanic_train['Embarked'].isnull(), titanic_train['Embarked'].mode(), titanic_train['Embarked'])
titanic_train.info()

# Age has around 160 missing values that can be filled with mean
titanic_train[titanic_train['Age'].isnull()]= titanic_train['Age'].mean()
titanic_train.info()

