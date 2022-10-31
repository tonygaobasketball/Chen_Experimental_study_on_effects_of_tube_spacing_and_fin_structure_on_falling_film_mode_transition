#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:12:44 2022

Title:
Experimental study on the effects of tube spacing and fin 
structure on falling film mode transitions with three-dimensional 
finned tubes
Authors:
J. Chen, Z. Gao

All rights reserved
"""

# Inputs:  (Change the values)
Ga = 194481
s_div_xi = 4.556
F = 28.00
# Select which model to use:
model_select = 'CS_S.model'


#---------------------
# Please do not change this part.
X_te = np.array([[Ga**0.25, s_div_xi, F]])
# Load the model:
RF = joblib.load(model_select)  
# Predict the Re value.  
y_hat = funcs.pred_KSVR(X_te, RF)
print('The predicted Re value is: {}'.format(y_hat))
#---------------------
