import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
import numpy as np
from sklearn.preprocessing import StandardScaler
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import backtesting_forecaster_multivariate
from skforecast.model_selection_multiseries import random_search_forecaster_multivariate


def evaluate_model_then_forecast_multivariate(
    df_arg,
    exog,
    lag_value,
    steps_value,
    freq,
    forecast_method="without_refit",
):
    """
    Function to perform time series forecasting using a DecisionTreeRegressor,
    optimize hyperparameters using random search, and evaluate the model using backtesting.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index and a single column of time series data.

    Returns:
    dict: Dictionary containing the best hyperparameters and evaluation metrics (MAE, MAPE, MSE, RMSE).
    """

    df = df_arg.copy(deep=True)
    df = df.reset_index()
    df = df.drop(df.columns[0], axis=1)

    # Initialize the forecaster with DecisionTreeRegressor
    forecaster = ForecasterAutoregMultiVariate(
        regressor=DecisionTreeRegressor(random_state=123),
        level=df.columns[-1],
        lags=lag_value,
        steps=10,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Define parameter grid to search for DecisionTreeRegressor
    param_grid = {
        "max_depth": [3, 5, 10, None],  # Depth of the tree
        "min_samples_split": [2, 5, 10],  # Minimum samples to split a node
        "min_samples_leaf": [1, 2, 4],  # Minimum samples per leaf
        "max_features": [
            None,
            "sqrt",
            "log2",
        ],  # Number of features to consider for split
    }

    # Perform random search to find the best hyperparameters
    results_random_search = random_search_forecaster_multivariate(
        forecaster=forecaster,
        series=df,  # The column of time series data
        param_distributions=param_grid,
        lags_grid=[3, 5, 7, 12, 14],
        steps=10,
        exog=exog,
        n_iter=10,
        metric="mean_squared_error",
        initial_train_size=int(len(df) * 0.8),
        fixed_train_size=False,
        return_best=True,  # Return the best parameter set
        random_state=123,
    )

    best_params = results_random_search.iloc[0]["params"]
    best_lag = int(max(list(results_random_search.iloc[0]["lags"])))
    # Recreate the forecaster with the best parameters
    forecaster = ForecasterAutoregMultiVariate(
        regressor=DecisionTreeRegressor(**best_params, random_state=123),
        level=df.columns[-1],
        lags=best_lag,
        steps=10,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Backtest the model
    backtest_metric, predictions = backtesting_forecaster_multivariate(
        forecaster=forecaster,
        series=df,
        steps=10,
        metric="mean_squared_error",
        initial_train_size=int(len(df) * 0.8),  # 80% train size
        levels=df.columns[-1],
        exog=exog,
        fixed_train_size=False,
        verbose=True,
    )

    y_true = df.iloc[int(len(df) * 0.8) :, -1]  # The actual values from the test set
    mae = mean_absolute_error(y_true, predictions)
    mape_val = mean_absolute_percentage_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)

    # Print evaluation metrics
    print(f"MAE: {mae}")
    print(f"MAPE: {mape_val}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    forecaster.fit(series=df)
    pred = forecaster.predict(steps=steps_value)

    # Generate index for the predicted values
    # extract the last datetime index index from df_arg
    last_index = df_arg.index[-1]
    # the result of this is just a date without time. While the function that take into account
    # the occurence of gap uses DateTimeIndex
    new_indices = pd.date_range(start=last_index, periods=steps_value + 1, freq=freq)[
        1:
    ]
    # new_indices = pd.to_datetime(new_indices)

    # Create a new DataFrame of the result
    forecast_df = pd.DataFrame(
        data=pred.values, index=new_indices, columns=[df.columns[-1]]
    )

    # Return results as a dictionary
    return {
        "results_random_search": results_random_search,
        "best_params": best_params,
        "mae": mae,
        "mape": mape_val,
        "mse": mse,
        "rmse": rmse,
        "pred_out": forecast_df,
        "pred_test": predictions,
    }
