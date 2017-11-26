#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:51:51 2017

@author: heisenberg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    global dataset, X, Y
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:,2:4].values
    Y = dataset.iloc[:, -1].values
    
def scale_data():
    global sc_x, X
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    X = sc_x.fit_transform(X)
    
def data_split():
    global x_train, x_test, y_train, y_test
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

def create_train_classifier():
    global classifier
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(x_train, y_train)
    
def predict():
    global y_pred
    y_pred = classifier.predict(x_test)
    print(y_pred)

def confusion_matrix_analysis():
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
def plot_graph():
    from  matplotlib.colors import ListedColormap
    x_set, y_set = x_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = min(x_set[:,0])-1, stop = max(x_set[:,0])+1, step = 0.01),
                         np.arange(start = min(x_set[:,1])-1, stop = max(x_set[:,0])+1, step = 0.01),)
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red','green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1], 
                    c=ListedColormap(('red','green'))(i), label=j)
    plt.title("KNN Regression")
    plt.xlabel("Age")
    plt.ylabel("Salary")
    plt.legend()
    #plt.show()

load_dataset()
scale_data()
data_split()
create_train_classifier()
predict()
confusion_matrix_analysis()
plot_graph()