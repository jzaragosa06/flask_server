
import pandas as pd
import numpy as np
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# from models.stacking import *
from models.stacking_multi import *
from utility.gap_functions import * 

# def forecast_multi_with_gap(df_arg, lag_list, steps_value, freq, gap_length, interval_length_before_gap, forecast_method='without_refit'):
def forecast_multi_with_gap(df_arg, dict_lags, steps_value, freq, gap_length, interval_length_before_gap, forecast_method='without_refit'):
    df = df_arg.copy(deep = True)

    #we'll just use the corresponding row number as index. 
    df = df.reset_index()
    df = df.drop(df.columns[0], axis = 1)
    
    # dict_lags = {}
    # #dictionary of lags
    # for i in range(len(df.columns)):
    #     dict_lags[df.columns[i]] =int( lag_list[i])

    #extract the last column as our target. 
    # last_col = df.columns[-1]
    
    stacking_regressor = build_stacking_regressor_multi(df_arg=df_arg, dict_lags=dict_lags)

    if forecast_method == 'without_refit':
        forecaster = ForecasterAutoregMultiVariate(
            regressor=stacking_regressor, 
            level=df.columns[-1], 
            lags=dict_lags, 
            # lags=lag_list,
            steps=steps_value, 
            transformer_series=StandardScaler(),
            transformer_exog=None,
            weight_func=None,
        )

        forecaster.fit(series = df)
        
        # Predict
        pred = forecaster.predict(steps=steps_value)


        # # Generate index for the predicted values
        # #extract the last datetime index index from df_arg
        # last_index = df_arg.index[-1]
        # #the result of this is just a date without time. While the function that take into account 
        # #the occurence of gap uses DateTimeIndex
        # new_indices = pd.date_range(
        #     start=last_index, periods=steps_value + 1, freq=freq
        # )[1:]
        # # new_indices = pd.to_datetime(new_indices)

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

        # Create a new DataFrame of the result
        #we can access the generated_indices
        generated_indices = pd.DatetimeIndex(generated_indices)
        
        #dataframe of the result
        forecast_df = pd.DataFrame(data = pred.values, index = generated_indices, columns=["target"])
        return forecast_df


# def evaluate_model_multi_with_gap(df_arg, lag_list, steps_value, freq, gap_length, interval_length_before_gap,forecast_method='without_refit'):
def evaluate_model_multi_with_gap(df_arg, dict_lags, steps_value, freq, gap_length, interval_length_before_gap,forecast_method='without_refit'):

    df = df_arg.copy(deep = True)

    # Ensure the DatetimeIndex has a frequency
    # df = df.asfreq(freq)
    #we'll just use the corresponding row number as index. 
    df = df.reset_index()
    df = df.drop(df.columns[0], axis = 1)

    # dict_lags = {}
    # #dictionary of lags
    # for i in range(len(df.columns)):
    #     dict_lags[df.columns[i]] =int( lag_list[i])

    test_size = 0.2
    test_samples = int(test_size * len(df))
    train_data, test_data = df.iloc[:-test_samples], df.iloc[-test_samples:]


    # Get the model
    stacking_regressor = build_stacking_regressor_multi(df_arg=df_arg, dict_lags=dict_lags)

    #then we need to built the forecaster. i.e., convert the regression model into forecasting model.
    if forecast_method == "without_refit":
        forecaster = ForecasterAutoregMultiVariate(
            regressor=stacking_regressor, 
            level=df.columns[-1], 
            lags=dict_lags, 
            # lags=lag_list,
            steps=steps_value, 
            transformer_series=StandardScaler(),
            transformer_exog=None, 
            weight_func=None, 
        )

        metric, predictions = backtesting_forecaster_multiseries(
                                           forecaster         = forecaster,
                                           series             = df,
                                           steps              = steps_value,
                                           metric             = 'mean_absolute_error',
                                           initial_train_size = len(train_data),
                                           refit              = False,
                                           fixed_train_size   = False,
                                           verbose            = False
        )

        # #then we generate the index. 
        # # Generate index for the predicted values
        # #number of last index on the train data. From there, we build the index of the test data. 
        # last_index = df_arg.index[len(train_data) -1]
        # #the result of this is just a date without time. While the function that take into account 
        # #the occurence of gap uses DateTimeIndex
        # new_indices = pd.date_range(
        #     start=last_index, periods=steps_value + 1, freq=freq
        # )[1:]
        # # new_indices = pd.to_datetime(new_indices)

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
        len_test_data = len(test_data)

        for _ in range(len_test_data):
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
            data=predictions.values, index=generated_indices, columns=["target"]
        )

        return metric, forecast_df


        
    else:
        ...



    



