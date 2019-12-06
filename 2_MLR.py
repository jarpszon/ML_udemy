# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt

#import dataset
dataset = pd.read_csv('2_MLR_50_Startups.csv')
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

# Visualisation
plt.pyplot.plot(y_pred) #, colour='red')
plt.pyplot.plot(y_test)
plt.pyplot.show()

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as smOLS
# adding column with 1s to the X 
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1 )
X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = smOLS.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = smOLS.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,5]]
regressor_OLS = smOLS.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = smOLS.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = smOLS.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()







