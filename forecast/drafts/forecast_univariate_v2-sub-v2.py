import pandas as pd
import numpy as np

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.datasets import fetch_dataset

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# this is relative import
# from seasonality_analysis import *
from models.stacking import *

from sklearn.model_selection import train_test_split


def forecast_uni(df_arg, lag_value, steps_value, freq, forecast_method="without_refit"):
    df = df_arg.copy(deep = True)

    # Ensure the DatetimeIndex has a frequency
    #this will fill the intermediate index. We shall not use this.
    # df = df.asfreq(freq)

    #we'll just use the corresponding row number as index. 
    df = df.reset_index()
    df = df.drop(df.columns[0], axis = 1)

    # Get the model
    stacking_regressor = build_staking_regressor()

    if forecast_method == "without_refit":
        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )

        # Fit the model
        forecaster.fit(df.iloc[:, -1])

        # Predict
        pred = forecaster.predict(steps=steps_value)

        # Generate index for the predicted values
        last_index = df_arg.index[-1]
        #the result of this is just a date without time. While the function that take into account 
        #the occurence of gap uses DateTimeIndex
        new_indices = pd.date_range(
            start=last_index, periods=steps_value + 1, freq=freq
        )[1:]
        # new_indices = pd.to_datetime(new_indices)

        # Create a new DataFrame of the result
        forecast_df = pd.DataFrame(
            data=pred.values, index=new_indices, columns=["target"]
        )

    else:
        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )

        pred_values = []

        for i in range(steps_value):
            forecaster.fit(df.iloc[:, -1])

            pred_i = pd.DataFrame(forecaster.predict(steps=1))
            pred_i_value = pred_i.iloc[0, 0]
            pred_values.append(pred_i_value)

            #add the column value to the df
            df.loc[len(df)] = pred_i_value

        #then we generate the index. 
        # Generate index for the predicted values
        last_index = df_arg.index[-1]
        #the result of this is just a date without time. While the function that take into account 
        #the occurence of gap uses DateTimeIndex
        new_indices = pd.date_range(
            start=last_index, periods=steps_value + 1, freq=freq
        )[1:]
        # new_indices = pd.to_datetime(new_indices)

        # Create a new DataFrame of the result
        forecast_df = pd.DataFrame(
            data=pred_values, index=new_indices, columns=["target"]
        )
        #======================================================================================

    return forecast_df


def evaluate_model(
    df_arg, lag_value, steps_value, freq, forecast_method="without_refit"
):
    df = df_arg.copy(deep = True)

    # Ensure the DatetimeIndex has a frequency
    # df = df.asfreq(freq)
    #we'll just use the corresponding row number as index. 
    df = df.reset_index()
    df = df.drop(df.columns[0], axis = 1)

    test_size = 0.2
    test_samples = int(test_size * len(df))
    train_data, test_data = df.iloc[:-test_samples], df.iloc[-test_samples:]

    # Get the model
    stacking_regressor = build_staking_regressor()

    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
    )

    if forecast_method == "without_refit":
        # Metrics and predictions are on the testing
        metric, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=df.iloc[:, -1],
            initial_train_size=len(train_data),
            steps=steps_value,
            refit=False,
            fixed_train_size=True,
            metric="mean_squared_error",
            n_jobs="auto",
            verbose=False,
        )
    else:
        # Metrics and predictions are on the testing
        metric, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=df.iloc[:, -1],
            initial_train_size=len(train_data),
            steps=steps_value,
            refit=True,
            fixed_train_size=False,
            metric="mean_squared_error",
            n_jobs="auto",
            verbose=False,
        )

    #then we generate the index. 
    # Generate index for the predicted values
    #number of last index on the train data. From there, we build the index of the test data. 
    last_index = df_arg.index[len(train_data) -1]
    #the result of this is just a date without time. While the function that take into account 
    #the occurence of gap uses DateTimeIndex
    new_indices = pd.date_range(
        start=last_index, periods=steps_value + 1, freq=freq
    )[1:]
    # new_indices = pd.to_datetime(new_indices)

    # Create a new DataFrame of the result
    forecast_df = pd.DataFrame(
        data=predictions.values, index=new_indices, columns=["target"]
    )

    return metric, predictions



    

