# -*- coding: utf-8 -*-
"""
Created on 2018

@author: mengda
"""

# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

os.chdir("C:\E-Work\Coursera\MLforDataSci")


#Load the dataset

AH_data = pd.read_csv("adult.csv")

data_clean = AH_data.dropna()

data_clean


#Convert string to number for decision tree training

from sklearn import preprocessing

number = preprocessing.LabelEncoder()
data_clean['native_country'] = number.fit_transform(data_clean.native_country)
data_clean['sex'] = number.fit_transform(data_clean.sex)
data_clean['race'] = number.fit_transform(data_clean.race)
data_clean['relationship'] = number.fit_transform(data_clean.relationship)
data_clean['occupation'] = number.fit_transform(data_clean.occupation)
data_clean['marital_status'] = number.fit_transform(data_clean.marital_status)
data_clean['education'] = number.fit_transform(data_clean.education)
data_clean['workclass'] = number.fit_transform(data_clean.workclass)

data_clean



#Split into training and testing sets

predictors = data_clean[['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                         'marital_status', 'occupation', 
                         'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                         'hours_per_week', 'native_country']]

targets = data_clean.income

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)


#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)
sklearn.metrics.accuracy_score(tar_test, predictions)


#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())




