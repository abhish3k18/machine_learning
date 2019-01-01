#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:11:36 2017

@author: heisenberg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def data_preparation(file_name):
    global X, Y, dataset
    #dataset = pd.read_csv(file_name)
    dataset = pd.read_excel(file_name)
    X = dataset.iloc[:,1:].values
    X = X[:,:-1]
    Y = dataset.iloc[:,0].values
    '''
    #ENCODING STATE VALUES
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    X [:,3] = LabelEncoder().fit_transform(X[:,3])
    X = OneHotEncoder(categorical_features=[3]).fit_transform(X).toarray()
    #X = OneHotEncoder(categories='auto').fit_transform(X).toarray()

    #AVOIDING DUMMY VARIABLE TRAP
    X = X[:, 1:]
	'''
def data_split():
    global x_train,x_test, y_train, y_test
    #TRAIN TEST SPLIT
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)
    
def train_model():
    global multiple_linear_regressor
    from sklearn.linear_model import LinearRegression
    multiple_linear_regressor = LinearRegression()
    #TRAIN MODEL
    multiple_linear_regressor.fit(x_train, y_train)
    return multiple_linear_regressor
    
def save_Regressor(regressor):
	with open('mL_Regressor.pickle','wb') as f:
		pickle.dump(regressor,f)

def load_regressor(pickle_Name):
	loaded_pickle=open(pickle_Name,'rb')		
	regressor=pickle.load(loaded_pickle)
	return regressor

def predict(regressor):
    global y_pred
    y_pred = regressor.predict(x_test)
    print(y_pred)
    print(y_test)

def backward_elimination():
    #PERFORMING BACKWARD ELIMINATION
    import statsmodels.formula.api as sm
    x_new = np.append(arr = np.ones((10,1)), values = X, axis=1)
    print(x_new)
    bE_regressor = sm.OLS(endog = Y, exog = x_new).fit()
    print(bE_regressor.rsquared)
    print(bE_regressor.summary())
    '''global bE_regressor
    from sklearn.linear_model import LinearRegression
    bE_regressor= LinearRegression()
    bE_regressor.fit(x_new, y_train)'''
    #CHECK p-value FOR EACH VARIABLE AND REMOVE ACCORDINGLY
    return bE_regressor
    
data_preparation("Hollywood.xls")
'''
Hollywood Movies

The data (X1, X2, X3, X4) are for each movie
X1 = first year box office receipts/millions
X2 = total production costs/millions
X3 = total promotional costs/millions
X4 = total book sales/millions'''

data_split()
#normal_regressor = train_model()
#save_Regressor(normal_regressor)
normal_regressor=load_regressor('mL_Regressor.pickle')
#backward_elimination()
predict(normal_regressor)

