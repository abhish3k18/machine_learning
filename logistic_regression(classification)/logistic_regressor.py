#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:55:29 2017

@author: heisenberg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    global dataset, X, Y
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:,2:4].values
    Y = dataset.iloc[:,4:5].values

def feature_scale():
    global X, Y, sc_x
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    X = sc_x.fit_transform(X)
    
def split_data():
    global x_train, x_test, y_train, y_test 
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
    
def create_train_regressor():
    global regressor
    from sklearn.linear_model import LogisticRegression
    regressor = LogisticRegression(random_state=0)
    regressor.fit(x_train, y_train)
    
def predict_values():
    pred = regressor.predict(x_test)
    print(pred)

def confusion_matrix_analysis():
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, regressor.predict(x_test))
    print(cm)
    
    

load_dataset()
feature_scale()
split_data()
create_train_regressor()
predict_values()
confusion_matrix_analysis()