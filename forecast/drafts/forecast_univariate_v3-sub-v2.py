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


#by default, it will be used to forecast the ts with no gap. 
def forecast_uni(df_arg, lag_value, steps_value, freq, forecast_method="without_refit"):
    df = df_arg.copy()

    # Ensure the DatetimeIndex has a frequency
    #WE'LL JUST COMMENT THIS OUT FOR NOW. 
    # df = df.asfreq(freq)

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
        last_index = df.index[-1]
        new_indices = pd.date_range(
            start=last_index, periods=steps_value + 1, freq=freq
        )[1:]

        # Create a new DataFrame of the result
        forecast_df = pd.DataFrame(
            data=pred.values, index=new_indices, columns=["target"]
        )

    else:
        temp_data = df.copy()
        last_col = df.columns[-1]

        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )

        pred_values = []
        pred_indices = []

        last_index = temp_data.index[-1]  # Properly define last_index here

        for i in range(steps_value):
            # Fit
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
            last_index = new_index  # Update last_index for the next iteration

            temp_data.loc[new_index, last_col] = pred_i.iloc[0, 0]

        pred_indices = pd.DatetimeIndex(pred_indices)
        # Store as a DataFrame
        forecast_df = pd.DataFrame(
            data=pred_values, index=pred_indices, columns=["target"]
        )


    return forecast_df

def forecast_uni_with_gap(df_arg, lag_value, steps_value, freq, gap_length, interval_length, forecast_method="without_refit"):
    df = df_arg.copy()

    # Ensure the DatetimeIndex has a frequency
    #WE'LL JUST COMMENT THIS OUT FOR NOW. 
    # df = df.asfreq(freq)

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



        #=========================================================================
        count_before = 0
        i = len(df) - 1
        temp = interval_length
        if freq == 'D':
            while i != 1 and temp != 0:
                diff = df.index[i] - df.index[i - 1]
                if (diff.days != 1):
                    ++count_before
                    break
                else:
                    ++count_before
                    --temp
        elif freq == 'W':
            while i != 1 and temp != 0:
                diff = df.index[i] - df.index[i - 1]
                if (diff.days != 7):
                    ++count_before
                    break
                else:
                    ++count_before
                    --temp
        #=========================================================================


        # we need to generate index for the predicted values.
        last_index = df.index[-1]
        new_indices = []

        temp_counter = count_before

        for i in range(1, steps_value + 1):
            if freq == "D":
                if temp_counter % interval_length == 0:
                    # new_index = last_index + pd.DateOffset(days=i) + pd.DateOffset(days=gap_length)
                    new_index = last_index + pd.DateOffset(days=gap_length)
                    temp_counter = 1
                else:
                    new_index = last_index + pd.DateOffset(days=1)
                    ++temp_counter
            elif freq == "W":
                if freq == "D":
                    if temp_counter % interval_length == 0:
                        # new_index = last_index + pd.DateOffset(days=i) + pd.DateOffset(days=gap_length)
                        new_index = last_index + pd.DateOffset(month=gap_length)
                        temp_counter = 1
                    else:
                        new_index = last_index + pd.DateOffset(month=1)
                        ++temp_counter
                # elif freq == "M":
                #     new_index = last_index + pd.DateOffset(months=i)
                # elif freq == "Q":
                #     new_index = last_index + pd.DateOffset(months=3 * i)
                # elif freq == "Y":
                #     new_index = last_index + pd.DateOffset(years=i)
            else:
                raise ValueError(f"Frequency '{freq}' is not supported")

            last_index = new_index
            new_indices.append(new_index)

        # converting the list to datetimeindex
        new_indices = pd.DatetimeIndex(new_indices)

        # Create a new DataFrame of the result
        forecast_df = pd.DataFrame(
            data=pred.values, index=new_indices, columns=["target"]
        )

    else:
        temp_data = df.copy()
        last_col = df.columns[-1]

        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )

        pred_values = []
        pred_indices = []

        last_index = temp_data.index[-1]  # Properly define last_index here

        for i in range(steps_value):
            # Fit
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
            last_index = new_index  # Update last_index for the next iteration

            temp_data.loc[new_index, last_col] = pred_i.iloc[0, 0]

        pred_indices = pd.DatetimeIndex(pred_indices)
        # Store as a DataFrame
        forecast_df = pd.DataFrame(
            data=pred_values, index=pred_indices, columns=["target"]
        )

    return forecast_df





















def evaluate_model(
    df_arg, lag_value, steps_value, freq, forecast_method="without_refit"
):
    df = df_arg.copy()

    # Ensure the DatetimeIndex has a frequency
    df = df.asfreq(freq)

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
        metric, prediction = backtesting_forecaster(
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
        metric, prediction = backtesting_forecaster(
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

    return metric, prediction
