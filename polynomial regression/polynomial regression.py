#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:22:10 2017

@author: heisenberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_preparation(file_name):
    global dataset, X, Y, poly_feat, x_poly
    dataset = pd.read_csv(file_name)
    #WE WONT BE TAKING POSITIONS, ONLY LEVELS, AS TAKING BOTH WOULD CREATE A REDUNDANCY
    # Do not use the following : X = dataset.iloc[:, 1].values [we need a 2-D array, and this command will create a 1-d array]
    X = dataset.iloc[:, 1:2].values
    Y = dataset.iloc[:, -1].values
    
    #FOR POLYNOMIAL REGRESSION WE NEED A DEGREE FEATURE (y = b0 + b1x +b2x^2 +b2x^3 +b2x^4)
    #WE HAVE X, NOW WE'LL CALCULATE IT'S DEGREES
    from sklearn.preprocessing import PolynomialFeatures
    poly_feat = PolynomialFeatures(degree=4)
    x_poly = poly_feat.fit_transform(X) #NOW X HAS THE POLYNOMIAL COMPONENTS REQUIRED TO RUN POLYNOMIAL REGRESSION

def data_split():
    global x_train, x_test, y_train, y_test
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_poly, Y, test_size = 0.2)

def train_model():
    global regressor
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_poly,Y)
    
def prediction():
    print(regressor.predict(poly_feat.transform(6.5))) #PREDICTING A VALUE OF 6.5
    
def visualization():
    plt.scatter(X, Y, color='red')
    #TO INCREASE GRAPH SMOOTHNESS
    x_grid = np.arange(min(X), max(X), 0.01) #creates a new array starting from minimum of X to maximum of X with an interval of 0.01
    x_grid = x_grid.reshape(len(x_grid), 1) #converting the array to matrix where row = number of rows in x_grid, column = 1
    plt.plot(x_grid, regressor.predict(poly_feat.transform(x_grid)), color='blue')
    plt.title("Polynomial Regression")
    plt.show()
    
data_preparation('Position_Salaries.csv')
#data_split() AS WE HAVE DON'T HAVE HIGH AMOUNT OF DATA HENCE WE'LL NOT SPLIT THE DATA
train_model()
prediction()
visualization()