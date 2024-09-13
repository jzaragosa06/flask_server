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

from sklearn.model_selection import train_test_split

# this is relative import
# from seasonality_analysis import *
from models.stacking import *


def forcast_uni(df_arg, lag_value, steps_value, freq, forecast_method="without_refit"):
    df = df_arg.copy(deep = True)
    # get the model
    stacking_regressor = build_staking_regressor()

    if forecast_method == "without_refit":
        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )

        # fit the model
        forecaster.fit(df.iloc[:, -1])

        # predict
        pred = forecaster.predict(steps=steps_value)

        # we need to handle the gaps
        # we will solve it later

        # we need to generate index for the predicted values.
        last_index = df.index[-1]
        new_indices = []

        # we add 1 since the range is exclusive on the upperbound.
        for i in range(1, steps_value + 1):
            if freq == "D":
                new_index = last_index + pd.DateOffset(days=i)
            elif freq == "W":
                new_index = last_index + pd.DateOffset(weeks=i)
            elif freq == "M":
                new_index = last_index + pd.DateOffset(months=i)
            elif freq == "Q":
                new_index = last_index + pd.DateOffset(months=3 * i)
            elif freq == "Y":
                new_index = last_index + pd.DateOffset(years=i)
            else:
                raise ValueError(f"Frequency '{freq}' is not supported")

            new_indices.append(new_index)

        # converting the list to datetimeindex
        new_indices = pd.DatetimeIndex(new_indices)

        # then create a new dataframe of result
        forecast_df = pd.DataFrame(
            data=pred.values, index=new_indices, columns=["target"]
        )
    else:

        temp_data = df.copy(deep = True)
        last_col = df.columns[-1]

        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )

        pred_values = []
        pred_indices = []

        for i in range(steps_value):
            # fit
            forecaster.fit(temp_data.iloc[:, -1])

            pred_i = pd.DataFrame(forecaster.predict(steps=1))
            pred_values.append(pred_i.iloc[0, 0])

            if freq == "D":
                new_index = last_index + pd.DateOffset(days=1)
            elif freq == "W":
                new_index = last_index + pd.DateOffset(weeks=1)
            elif freq == "M":
                new_index = last_index + pd.DateOffset(months=1)
            elif freq == "Q":
                new_index = last_index + pd.DateOffset(months=3)
            elif freq == "Y":
                new_index = last_index + pd.DateOffset(years=1)
            else:
                raise ValueError(f"Frequency '{freq}' is not supported")

            pred_indices.append(new_index)

            temp_data.loc[pd.DatetimeIndex(new_index), last_col] = pred_i.iloc[0, 0]

        pred_indices = pd.DatetimeIndex(pred_indices)
        # store as a dataframe
        forecast_df = pd.DataFrame(
            data=pred_values, index=pred_indices, columns=["target"]
        )


def evaluate_model(
    df_arg, lag_value, steps_value, freq, forecast_method="without_refit"
):
    df = df_arg.copy(deep = True)

    test_size = 0.2
    test_samples = int(test_size * len(df))
    train_data, test_data = df[:-test_samples], df[-test_samples]

    # get the model
    stacking_regressor = build_staking_regressor()

    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
    )

    if forecast_method == "without_refit":
        # metrics and predictions are on the testing
        metric, prediction = backtesting_forecaster(
            forecaster=forecaster,
            y=df.iloc[:-1],
            initial_train_size=len(train_data),
            steps=12,
            refit=False,
            fixed_train_size=True,
            metric="mean_squared_error",
            n_jobs="auto",
            verbose=False,
        )

        return metric, prediction
    else:
        # metrics and predictions are on the testing
        metric, prediction = backtesting_forecaster(
            forecaster=forecaster,
            y=df.iloc[:-1],
            initial_train_size=len(train_data),
            steps=12,
            refit=True,
            fixed_train_size=False,
            metric="mean_squared_error",
            n_jobs="auto",
            verbose=False,
        )

        return metric, prediction
