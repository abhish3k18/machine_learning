#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 22:38:13 2017

@author: heisenberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
    
def data_preparation(file_name):
    global X, Y, x_test, x_train, y_test, y_train
    data = pd.read_csv('Salary_Data.csv') #LOAD DATASET
    X = data.iloc[:,:-1].values #GET THE DEPENDENT VARIABLE SET
    Y = data.iloc[:,-1].values #GET THE INDEPENDENT VARIABLE SET (to be predicted)
    #DIVIDE THE DATA INTO TEST AND TRAINING SETS
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

def create_regressor():
    global regressor
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    
def predict_data():
    y_pred = regressor.predict(x_test)
    print(y_pred)

def plot_chart():
    plt.scatter(X, Y, color='red')
    plt.plot(x_test, regressor.predict(x_test), color='Blue')
    plt.title("Salary vs Experience")
    plt.ylabel("Salary")
    plt.xlabel("Experience")
    plt.plot
    
data_preparation('Salary_Data.csv')
create_regressor()
plot_chart()
