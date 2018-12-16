# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 12:53:43 2018

@author: adkarumo
"""

import pandas as pd
import io
import pydot
from sklearn import tree

#os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

titanic_train= pd.read_csv("D:\\Users\\adkarumo\\Documents\\Data Science\\Titanic\\train.csv")

x_train= titanic_train[['Pclass','SibSp','Parch']]
y_train= titanic_train['Survived']

decision_tree= tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

# visualize the decision tree
string_IO= io.StringIO()
tree.export_graphviz(decision_tree, out_file= string_IO, feature_names= x_train.columns)

file= pydot.graph_from_dot_data(string_IO.getvalue())[0]
os.chdir("D:\\Users\\adkarumo\\Documents\\Data Science\\Titanic\\")
file.write_pdf("DecisionTree-1.pdf")

