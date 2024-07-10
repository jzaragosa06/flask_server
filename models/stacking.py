import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def build_staking_regressor(): 
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

