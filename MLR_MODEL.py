# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:18:45 2021

@author: Sayon Som

AIM: A VENTURE CAPITALIST FIRM IS TO PREDICT THE PROFIT OF A PARTICULAR STARTUP TO INVEST TO GAIN THE MAXIMUM PROFIT
"""

# Multiple Linear Regression
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
column=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],remainder="passthrough")
X=np.array(column.fit_transform(X))
print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
multireg=LinearRegression()
multireg.fit(X_train,y_train)
# Predicting the Test set results
y_pred=multireg.predict(X_test)
np.set_printoptions(precision=2)
print("Predicted \t\t Real")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 
#lets check for a specific training element
print("The training ex is {}".format(X_train[25]))
print("The predicted result is {}".format(multireg.predict([X_train[25]])))
 