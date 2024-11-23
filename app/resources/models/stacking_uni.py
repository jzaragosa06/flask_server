# """
# In this version:
# -We optimize the parameters to the base models before incorporating them in a stacking regressor model.
# -The time it takes to find the parameter to this model may vary for up to 10 minutes each.
# -This is fore uni. 
# """

# import pandas as pd
# import numpy as np

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.datasets import make_regression
# from sklearn.metrics import mean_squared_error

# from app.resources.utility.create_feature import *



# def optimize_base_models_uni(df_arg, lag_value):
#     # for this, we will use the following models as base:
#         # LinearRegression
#         # SVR
#         # RandomForestRegressor
#         # GradientBoostingRegressor
#         # The meta-model is
#         # LinearRegression

#     lagged_data = create_lag_features_uni(df_arg=df_arg, lag_value=lag_value)
#     # column name of the last col.

#     # separate x and y.
#     X = lagged_data.drop(df_arg.columns[-1], axis=1)
#     Y = lagged_data[df_arg.columns[-1]]

#     # we need to return a list of tuples.
#     base_estimators = [
#         ('lr', LinearRegression()),
#     ]

#     # we're defining parameter grids for each model
#     param_grids = {
#         'RandomForestRegressor': {
#             'n_estimators': [100, 200, 300],
#             'max_depth': [None, 10, 20, 30],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4]
#         },
#         'GradientBoostingRegressor': {
#             'n_estimators': [100, 200, 300],
#             'learning_rate': [0.01, 0.1, 0.2],
#             'max_depth': [3, 5, 7],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4]
#         },
#         'SVR': {
#             'svr__C': [0.1, 1, 10, 100],
#             'svr__epsilon': [0.01, 0.1, 1],
#             'svr__kernel': ['linear', 'rbf'],
#             'svr__gamma': ['scale', 'auto']  # Only relevant for 'rbf' kernel
#         }
#     }

#     # Define models
#     models = {
#         'RandomForestRegressor': RandomForestRegressor(),
#         'GradientBoostingRegressor': GradientBoostingRegressor(),
#         'SVR': SVR()
#     }

#     best_params = {}
#     for model_name, model in models.items():
#         print(f"Optimizing {model_name}...")
#         grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#         grid_search.fit(X, Y)
#         best_params[model_name] = grid_search.best_params_
#         params = grid_search.best_params_
#         print(f"Best parameters for {model_name}: {params}")

#         if model_name == 'SVR':
#             best_model = SVR(**params)
#             base_estimators.append(('svr', best_model))
#         elif model_name == 'RandomForestRegressor':
#             best_model = RandomForestRegressor(**params)
#             base_estimators.append(('rfr', best_model))
#         elif model_name == 'GradientBoostingRegressor':
#             best_model = GradientBoostingRegressor(**params)
#             base_estimators.append(('gbr', best_model))
    
#     return base_estimators
        
        

# def build_stacking_regressor_uni(df_arg, lag_value):
#     # # basically, this is a list of tuples, where each tuple contains  2 elements.
#     # base_estimators = optimize_base_models_uni(df_arg = df_arg, lag_value=lag_value)

#     # # Initialize stacking regressor with a linear regression meta-estimator
#     # stacking_regressor = StackingRegressor(
#     #     estimators=base_estimators,
#     #     final_estimator=LinearRegression()
#     # )

#     # return stacking_regressor

#     #=======================================================================================

#     base_estimators = [
#         ('lr', LinearRegression()),
#         ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
#         ('svr', SVR(kernel='sigmoid', C=100, gamma=0.5),),
#         ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, min_samples_split=5, min_samples_leaf=2))
        
#     ]

#     # Initialize stacking regressor with a linear regression meta-estimator
#     stacking_regressor = StackingRegressor(
#         estimators=base_estimators,
#         final_estimator=LinearRegression()
#     )

#     return stacking_regressor




import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster, backtesting_forecaster

def evaluate_ridge_and_lasso_enr_dt(df_arg, exog, lag_value):
    """
    Evaluate a time series forecasting model using a StackingRegressor
    with RandomForest, XGBoost, and Ridge, optimized with random search
    and evaluated using backtesting.
    """
    
    
    df = df_arg.copy(deep=True).reset_index(drop=True)

    # Define base and meta estimators for StackingRegressor
    base_estimators = [
        ("lasso", Lasso(random_state=123)),
        ("enr", ElasticNet(random_state=123)),
        ("dt", DecisionTreeRegressor(random_state=123)),
    ]
    meta_estimator = Ridge(random_state=123)
    stacking_regressor = StackingRegressor(
        estimators=base_estimators, final_estimator=meta_estimator
    )

    # Initialize the ForecasterAutoreg
    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor,
        lags=lag_value,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Define hyperparameter grid for random search
    param_grid = {
        "lasso__alpha": [0.001, 0.01, 0.1, 1, 10, 100], 
        "lasso__max_iter": [500, 1000, 1500], 
        "enr__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "enr__l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0],
        "dt__max_depth": [3, 5, 10, None],
        "dt__min_samples_split": [2, 5, 10], 
        "dt__min_samples_leaf":[1, 2, 4], 
        "dt__max_features":[None, 'sqrt', 'log2'],
        "final_estimator__alpha": [0.01, 0.1, 1, 10, 100],
        "final_estimator__fit_intercept": [True, False],
        "final_estimator__solver": ["auto", "svd", "cholesky", "lsqr", "saga"],
    }

    # Perform random search with verbose output
    search_results = random_search_forecaster(
        forecaster=forecaster,
        y=df.iloc[:, 0],  # Time series data
        param_distributions=param_grid,
        lags_grid=[3, 5, 7, 12, 14], 
        steps=10,
        exog=exog,
        n_iter=10,
        metric="mean_squared_error",
        initial_train_size=int(len(df) * 0.8),
        fixed_train_size=False,
        return_best=True,
        random_state=123,
        verbose=True,
    )
    
    print(search_results)

    # Extract best parameters
    best_params = search_results.iloc[0]["params"]
    lasso_params = {k.replace("lasso__", ""): v for k, v in best_params.items() if "lasso__" in k}
    enr_params = {k.replace("enr__", ""): v for k, v in best_params.items() if "enr__" in k}
    dt_params = {k.replace("dt__", ""): v for k, v in best_params.items() if "dt__" in k}
    ridge_params = {k.replace("final_estimator__", ""): v for k, v in best_params.items() if "final_estimator__" in k}
    
    best_lag =  int(max(list(search_results.iloc[0]["lags"])))

    # Recreate optimized StackingRegressor
    lasso_best = Lasso(**lasso_params)
    enr_best = ElasticNet(**enr_params)
    dt_best = DecisionTreeRegressor(**dt_params, random_state=123)
    ridge_best = Ridge(**ridge_params, random_state=123)
    stacking_regressor_best = StackingRegressor(
        estimators=[("lasso", lasso_best), ("enr", enr_best), ("dt", dt_best)], final_estimator=ridge_best
    )

    # Final ForecasterAutoreg
    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor_best,
        lags=best_lag,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Backtesting for evaluation
    backtest_metric, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=df.iloc[:, 0],
        exog=exog,
        initial_train_size=int(len(df) * 0.8),
        fixed_train_size=False,
        steps=10,
        metric="mean_squared_error",
        verbose=True,
    )

    # Compute evaluation metrics
    y_true = df.iloc[int(len(df) * 0.8) :, 0]
    mae = mean_absolute_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    
    
    #train the model on all the data
    forecaster.fit(df.iloc[:, -1])
    pred = forecaster.predict(steps = 12)
    
    # Display metrics
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # Return results
    return {
        "results_random_search": search_results,
        "best_params": best_params,
        "mae": mae,
        "mape": mape,
        "mse": mse,
        "rmse": rmse,
    }
