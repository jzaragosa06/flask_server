#In this version:
    #-We optimize the parameters to the base models before incorporating them in a stacking regressor model. 
    #-The time it takes to find the parameter to this model may vary for up to 10 minutes each. 
    #The meta model is one of the regularized regression models


import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def optimize_base_models():
    # for this, we will use the following models as base:
        #LinearRegression
        #SVR
        #RandomForestRegressor
        #GradientBoostingRegressor
    # The meta-model is 
        #LinearRegression
    ...


def build_staking_regressor(): 
    #basically, this is a list of tuples, where each tuple contains  2 elements. 
    base_estimators = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('svr', SVR(kernel='sigmoid', C=100, gamma=0.5),)
    ]

    # Initialize stacking regressor with a linear regression meta-estimator
    stacking_regressor = StackingRegressor(
        estimators=base_estimators,
        final_estimator=LinearRegression()
    )

    return stacking_regressor
