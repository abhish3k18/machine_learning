#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:11:36 2017

@author: heisenberg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_preparation(file_name):
    global X, Y, dataset
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values
    
    #ENCODING STATE VALUES
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    X [:,3] = LabelEncoder().fit_transform(X[:,3])
    X = OneHotEncoder(categorical_features=[3]).fit_transform(X).toarray()
    
    #AVOIDING DUMMY VARIABLE TRAP
    X = X[:, 1:]
    
def data_split():
    global x_train,x_test, y_train, y_test
    #TRAIN TEST SPLIT
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3)
    
def train_model():
    global multiple_linear_regressor
    from sklearn.linear_model import LinearRegression
    multiple_linear_regressor = LinearRegression()
    #TRAIN MODEL
    multiple_linear_regressor.fit(x_train, y_train)
    return multiple_linear_regressor
    
def predict(regressor):
    global y_pred
    y_pred = multiple_linear_regressor(x_test)

def backward_elimination():
    #PERFORMING BACKWARD ELIMINATION
    import statsmodels.formula.api as sm
    x_new = np.append(arr = np.ones((50,1)), values = X, axis=1)
    bE_regressor = sm.OLS(endog = Y, exog = x_new).fit()
    print(bE_regressor.rsquared)
    print(bE_regressor.summary())
    #CHECK p-value FOR EACH VARIABLE AND REMOVE ACCORDINGLY
    return bE_regressor
    
data_preparation("50_Startups.csv")
data_split()
normal_regressor = train_model()
backward_elimination_regressor = backward_elimination()
predict(normal_regressor)
predict(backward_elimination_regressor)