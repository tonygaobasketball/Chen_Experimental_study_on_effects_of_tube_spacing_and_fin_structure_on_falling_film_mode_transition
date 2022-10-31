# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:36:05 2022

Functions used for Chen's work.

@author: tonyg
"""

import numpy as np

def para_est(dt):
    # Re = A * Ga^0.25 * (s/xi)^b * P^c
    # input: dt (arrays) has features Re, Ga^0.25, s, xi, s/xi, P
    # Output: A, b, c.
    
    sub_dt = np.log(np.array(dt))
    Y = sub_dt[:,0] - sub_dt[:,1]
    temp = sub_dt[:,-2:]
    X = np.append(np.ones([temp.shape[0],1]), temp, axis = 1)
    # Apply least squares on Y and X.
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)
    beta = np.dot(np.linalg.inv(XTX), XTY)
    # Recover the parameters A, b and c.
    para_A = np.exp(beta[0])
    b = beta[1]
    c = beta[2]
    
    return para_A, b, c

def pred_para_est(x, para_A, b, c):
    # input: single data point x.
    # output: response value yhat.
    yhat = para_A*x[0]*(x[1])**b*(x[2])**c
    return yhat


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
def KSVR_rbf(X, y):
    # formulate SVR model with rbf kernel
    parameters = {'C':list(4**np.array(range(-4, 13), dtype = float)), 
                  'gamma': list(2**np.array(range(-4, 8), dtype = float)),
                  }
    svr_rbf = SVR(kernel="rbf")
    regressor = GridSearchCV(svr_rbf, parameters)
    regressor.fit(X, y)
    yhat = regressor.predict(X)
    best_model = regressor
    
    return yhat, best_model

def KSVR_p2(X, y):
    # formulate SVR model with quadratic kernel
    parameters = {'C':list(4**np.array(range(0, 1), dtype = float))}
    svr_p2 = SVR(kernel="poly", gamma = 'scale', degree = 2)
    regressor = GridSearchCV(svr_p2, parameters)
    regressor.fit(X, y)
    yhat = regressor.predict(X)
    best_model = regressor
    
    return yhat, best_model

from sklearn.neural_network import MLPRegressor
def MLP(X, y):
    # Multi-layer perceptron
    # parameters = {'hidden_layer_sizes': X.shape[1]*list(4**np.array(range(0, 8), dtype = float))}
    regressor = MLPRegressor(max_iter=300, random_state = 1)
    regressor.fit(X, y)
    yhat = regressor.predict(X)
    best_model = regressor
    
    return yhat, best_model


def pred_KSVR(Xte, te_model):
    ypred = te_model.predict(Xte)
    return ypred