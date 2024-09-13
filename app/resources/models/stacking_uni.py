"""
In this version:
-We optimize the parameters to the base models before incorporating them in a stacking regressor model.
-The time it takes to find the parameter to this model may vary for up to 10 minutes each.
-This is fore uni. 
"""


import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from app.resources.utility.create_feature import *



def optimize_base_models_uni(df_arg, lag_value):
    # for this, we will use the following models as base:
        # LinearRegression
        # SVR
        # RandomForestRegressor
        # GradientBoostingRegressor
        # The meta-model is
        # LinearRegression

    lagged_data = create_lag_features_uni(df_arg=df_arg, lag_value=lag_value)
    # column name of the last col.

    # separate x and y.
    X = lagged_data.drop(df_arg.columns[-1], axis=1)
    Y = lagged_data[df_arg.columns[-1]]

    # we need to return a list of tuples.
    base_estimators = [
        ('lr', LinearRegression()),
    ]

    # we're defining parameter grids for each model
    param_grids = {
        'RandomForestRegressor': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVR': {
            'svr__C': [0.1, 1, 10, 100],
            'svr__epsilon': [0.01, 0.1, 1],
            'svr__kernel': ['linear', 'rbf'],
            'svr__gamma': ['scale', 'auto']  # Only relevant for 'rbf' kernel
        }
    }

    # Define models
    models = {
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'SVR': SVR()
    }

    best_params = {}
    for model_name, model in models.items():
        print(f"Optimizing {model_name}...")
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, Y)
        best_params[model_name] = grid_search.best_params_
        params = grid_search.best_params_
        print(f"Best parameters for {model_name}: {params}")

        if model_name == 'SVR':
            best_model = SVR(**params)
            base_estimators.append(('svr', best_model))
        elif model_name == 'RandomForestRegressor':
            best_model = RandomForestRegressor(**params)
            base_estimators.append(('rfr', best_model))
        elif model_name == 'GradientBoostingRegressor':
            best_model = GradientBoostingRegressor(**params)
            base_estimators.append(('gbr', best_model))
    
    return base_estimators
        
        

def build_stacking_regressor_uni(df_arg, lag_value):
    # # basically, this is a list of tuples, where each tuple contains  2 elements.
    # base_estimators = optimize_base_models_uni(df_arg = df_arg, lag_value=lag_value)

    # # Initialize stacking regressor with a linear regression meta-estimator
    # stacking_regressor = StackingRegressor(
    #     estimators=base_estimators,
    #     final_estimator=LinearRegression()
    # )

    # return stacking_regressor

    #=======================================================================================

    base_estimators = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('svr', SVR(kernel='sigmoid', C=100, gamma=0.5),),
        ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, min_samples_split=5, min_samples_leaf=2))
        
    ]

    # Initialize stacking regressor with a linear regression meta-estimator
    stacking_regressor = StackingRegressor(
        estimators=base_estimators,
        final_estimator=LinearRegression()
    )

    return stacking_regressor
