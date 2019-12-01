# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:30:04 2019

@author: Użytkownik
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import matplotlib as plt

#import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#missing data
from  sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ="NaN", strategy="mean" , axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X=X[:,1:]

# spliting data into training set and test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediciting Test set result
y_pred = regressor.predict(X_test)













