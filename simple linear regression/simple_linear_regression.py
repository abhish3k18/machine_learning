#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 22:38:13 2017

@author: heisenberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv') #LOAD DATASET
X = data.iloc[:,:-1].values #GET THE DEPENDENT VARIABLE SET
Y = data.iloc[:,-1].values #GET THE INDEPENDENT VARIABLE SET (to be predicted)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


plt.scatter(X, Y, color='red')
plt.plot(x_test, regressor.predict(x_test), color='Blue')
plt.title("Salary vs Experience")
plt.ylabel("Salary")
plt.xlabel("Experience")
plt.plot
