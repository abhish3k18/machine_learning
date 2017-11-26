#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:06:19 2017

@author: heisenberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_dataset():
    global dataset, X, Y
    dataset = pd.read_csv("Position_Salaries.csv")
    X = dataset.iloc[:,1:2].values
    Y = dataset.iloc[:,2:3].values

# NO NEED TO SCALE THE DATA
    
def create_train_regressor():
    global regressor
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor()
    regressor.fit(X,Y)
    
def predict(value):
    pred = regressor.predict(value)
    print(pred)
    
def plot_graph():
    plt.scatter(X, Y, color = 'red')
    x_grid = np.arange(min(X), max(X), 0.01)
    x_grid = x_grid.reshape(len(x_grid), 1)
    plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
    plt.show()
    
load_dataset()
create_train_regressor()
predict(6.5)
plot_graph()