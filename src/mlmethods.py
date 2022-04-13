r"""
The functions that implement the various methods for incorporating theory
into machine learning. Benchmark methods of direct (no theory) and theory
(no machine learning) are also included. Specifically, the code below works
with ML models from scikit-learn such as their Random Forest Regressor
"""

import numpy as np
import random
import scipy

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def direct(x_train,y_train,t_train,a_train,x_test,y_test,t_test,mlmodel):
    """
    Run the machine learning model with no knowledge of theory and return
    the mean squared error.
    """
    
    mlmodel.fit(x_train, y_train)
    y_pred = mlmodel.predict(x_test)
    
    return mean_squared_error(y_test, y_pred)

def latentvariable(x_train,y_train,t_train,a_train,x_test,y_test,t_test,mlmodel):
    """
    Use numerical values of the theory as an additional input to the machine
    learning model. Return the mean squared error.
    """
    
    x_m_train=np.concatenate((x_train,t_train.reshape(len(x_train[:,0]),1)),1)
    x_m_test=np.concatenate((x_test,t_test.reshape(len(x_test[:,0]),1)),1)
    mlmodel.fit(x_m_train, y_train)
    y_pred = mlmodel.predict(x_m_test)
    
    return mean_squared_error(y_test, y_pred)

def multitask(x_train,y_train,t_train,a_train,x_test,y_test,t_test,mlmodel):
    """
    Use numerical values of the theory as an additional output to the machine
    learning model. Return the mean squared error.
    """

    y_m_train=np.concatenate((y_train.reshape(len(x_train[:,0]),1),t_train.reshape(len(x_train[:,0]),1)),1)
    y_m_test=np.concatenate((y_test.reshape(len(x_test[:,0]),1),t_test.reshape(len(x_test[:,0]),1)),1)
    mlmodel.fit(x_train, y_m_train)
    y_pred = mlmodel.predict(x_test)

    return mean_squared_error(y_test, y_pred[:,0])

def difference(x_train,y_train,t_train,a_train,x_test,y_test,t_test,mlmodel):
    """
    Use machine learning to predict the difference between the training data
    and numerical values of the theory. Then use that machine learning model
    to predict the difference at new input values. Next add the difference back
    to the predicted theory to yield a prediction for the output. Return the 
    mean squared error.
    """

    y_mod = y_train-t_train
    mlmodel.fit(x_train, y_mod)
    y_pred = mlmodel.predict(x_test) + t_test
    
    return mean_squared_error(y_test, y_pred)

def quotient(x_train,y_train,t_train,a_train,x_test,y_test,t_test,mlmodel):
    """
    Use machine learning to predict the quotient between the training data
    and numerical values of the theory. Then use that machine learning model
    to predict the quotient at new input values. Next multiply the quotient
    back to the predicted theory to yield a prediction for the output. Return
    the mean squared error.
    """

    y_mod = y_train/t_train
    mlmodel.fit(x_train, y_mod)
    y_pred = mlmodel.predict(x_test) * t_test
    
    return mean_squared_error(y_test, y_pred)

def theory(x_train,y_train,t_train,a_train,x_test,y_test,t_test):
    """
    Use theory to predict the output without any training data. Return the
    mean squared error.
    """

    return mean_squared_error(y_test, t_test)
