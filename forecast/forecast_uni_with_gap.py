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

from models.stacking_uni import *
from utility.gap_functions import *

from sklearn.model_selection import train_test_split



def forecast_uni_with_gap(df_arg, lag_value, steps_value, freq, gap_length, interval_length_before_gap, forecast_method="without_refit"):
    df = df_arg.copy(deep = True)
    colname = df.columns[0]
    stacking_regressor = build_stacking_regressor_uni(
    df_arg=df_arg, lag_value=lag_value)

    if forecast_method == "without_refit":
        #reset the index of the df to integer. 
        df = df.reset_index()
        df = df.drop(df.columns[0], axis = 1)

        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )

        # Fit the model
        forecaster.fit(df.iloc[:, -1])

        # Predict
        pred = forecaster.predict(steps=steps_value)

        #=========================================================================    
        count_before = compute_count_before(df_arg, freq, interval_length_before_gap)
        
        print(f"count_before: {count_before}")
        
        # Generate index for the predicted values
        #extract the index from the df_arg, since we revert the index to integer. 
        last_index = df_arg.index[-1]
        generated_indices = []
        current_index = last_index
        interval_counter = count_before

        for _ in range(steps_value):
            if interval_counter == interval_length_before_gap:
                if freq == 'D':
                    current_index += pd.DateOffset(days=gap_length + 1)
                elif freq == 'W':
                    current_index += pd.DateOffset(weeks=gap_length)
                elif freq == 'M':
                    current_index += pd.DateOffset(months=gap_length + 1)
                elif freq == 'Q':
                    current_index += pd.DateOffset(months=(gap_length + 1) * 3)
                elif freq == 'Y':
                    current_index += pd.DateOffset(years=gap_length + 1)
                interval_counter = 0
            else:
                if freq == 'D':
                    current_index += pd.DateOffset(days=1)
                elif freq == 'W':
                    current_index += pd.DateOffset(weeks=1)
                elif freq == 'M':
                    current_index += pd.DateOffset(months=1)
                elif freq == 'Q':
                    current_index += pd.DateOffset(months=3)
                elif freq == 'Y':
                    current_index += pd.DateOffset(years=1)
            interval_counter += 1
            generated_indices.append(current_index)
        #=========================================================================
        #we can access the generated_indices
        generated_indices = pd.DatetimeIndex(generated_indices)
        
        #dataframe of the result
        forecast_df = pd.DataFrame(data = pred.values, index = generated_indices, columns=[f"{colname}"])
        return forecast_df

    else:
        #reset the index of the df to integer. 
        df = df.reset_index()
        df = df.drop(df.columns[0], axis = 1)

        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )             
        
        pred_values = []
        
        for i in range(steps_value):
            forecaster.fit(df.iloc[:,-1])
            
            pred_i = pd.DataFrame(forecaster.predict(steps = 1))
            pred_i_value = pred_i.iloc[0,0]
            pred_values.append(pred_i_value)

            #add the value to the df
            df.loc[len(df)] = pred_i_value

        #=========================================================================    
        count_before = compute_count_before(df_arg, freq, interval_length_before_gap)
        
        print(f"count_before: {count_before}")
        
        # Generate index for the predicted values
        #extract the index from the df_arg, since we revert the index to integer. 
        last_index = df_arg.index[-1]
        generated_indices = []
        current_index = last_index
        interval_counter = count_before

        for _ in range(steps_value):
            if interval_counter == interval_length_before_gap:
                if freq == 'D':
                    current_index += pd.DateOffset(days=gap_length + 1)
                elif freq == 'W':
                    current_index += pd.DateOffset(weeks=gap_length + 1)
                elif freq == 'M':
                    current_index += pd.DateOffset(months=gap_length + 1)
                elif freq == 'Q':
                    current_index += pd.DateOffset(months=(gap_length + 1) * 3)
                elif freq == 'Y':
                    current_index += pd.DateOffset(years=gap_length + 1)
                interval_counter = 0
            else:
                if freq == 'D':
                    current_index += pd.DateOffset(days=1)
                elif freq == 'W':
                    current_index += pd.DateOffset(weeks=1)
                elif freq == 'M':
                    current_index += pd.DateOffset(months=1)
                elif freq == 'Q':
                    current_index += pd.DateOffset(months=3)
                elif freq == 'Y':
                    current_index += pd.DateOffset(years=1)
            interval_counter += 1
            generated_indices.append(current_index)
        #=========================================================================
        #we can access the generated_indices
        generated_indices = pd.DatetimeIndex(generated_indices)

        forecast_df = pd.DataFrame(data = pred_values, index = generated_indices, columns=[f"{colname}"])
        return forecast_df



def evaluate_model_uni_with_gap(
    df_arg, lag_value, steps_value, freq,  gap_length, interval_length_before_gap, forecast_method="without_refit"
):
    #We will use the  fifth and sixth argument to build the index. 
    df = df_arg.copy(deep = True)
    colname = df.columns[0]

    # Ensure the DatetimeIndex has a frequency
    # df = df.asfreq(freq)
    #we'll just use the corresponding row number as index. 
    df = df.reset_index()
    df = df.drop(df.columns[0], axis = 1)

    test_size = 0.2
    test_samples = int(test_size * len(df))
    train_data, test_data = df.iloc[:-test_samples], df.iloc[-test_samples:]

    # Get the model
    stacking_regressor = build_stacking_regressor_uni(
        df_arg=df_arg, lag_value=lag_value)

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

    #Generate index. Consider the gaps in the data. 
    #=========================================================================    
    count_before = compute_count_before(df_arg, freq, interval_length_before_gap)
    
    print(f"count_before: {count_before}")
    
    # Generate index for the predicted values
    #extract the index from the df_arg, since we revert the index to integer. 
    #the first part extracts the training data. the second part extracts the last index. 
    last_index = df_arg.iloc[:-test_samples].index[-1]
    generated_indices = []
    current_index = last_index
    interval_counter = count_before
    len_train_data = len(test_data)

    for _ in range(len_train_data):
        if interval_counter == interval_length_before_gap:
            if freq == 'D':
                current_index += pd.DateOffset(days=gap_length + 1)
            elif freq == 'W':
                current_index += pd.DateOffset(weeks=gap_length)
            elif freq == 'M':
                current_index += pd.DateOffset(months=gap_length + 1)
            elif freq == 'Q':
                current_index += pd.DateOffset(months=(gap_length + 1) * 3)
            elif freq == 'Y':
                current_index += pd.DateOffset(years=gap_length + 1)
            interval_counter = 0
        else:
            if freq == 'D':
                current_index += pd.DateOffset(days=1)
            elif freq == 'W':
                current_index += pd.DateOffset(weeks=1)
            elif freq == 'M':
                current_index += pd.DateOffset(months=1)
            elif freq == 'Q':
                current_index += pd.DateOffset(months=3)
            elif freq == 'Y':
                current_index += pd.DateOffset(years=1)
        interval_counter += 1
        generated_indices.append(current_index)
    #=========================================================================

    #then we convert the list to datetimeindex. 
    generated_indices = pd.DatetimeIndex(generated_indices)

    # Create a new DataFrame of the result
    forecast_df = pd.DataFrame(
        data=predictions.values, index=generated_indices, columns=[f"{colname}"]
    )

    return metric, forecast_df



    

