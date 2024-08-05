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

def compute_count_before(df, freq, interval_length_before_gap):
    count_before = 0
    temp = interval_length_before_gap
    i = len(df) - 1

    while i > 0 and temp > 0:
        diff = df.index[i] - df.index[i - 1]

        if freq == 'D' and diff.days != 1:
              count_before += 1
              break
        elif freq == 'W' and diff.days != 7:
              count_before += 1
              break
        elif freq == 'M' and (df.index[i].month == df.index[i - 1].month or diff.days > 31):
              count_before += 1
              break
        elif freq == 'Q' and (df.index[i].quarter == df.index[i - 1].quarter or diff.days > 93):
              count_before += 1
              break
        elif freq == 'Y' and df.index[i].year == df.index[i - 1].year:
              count_before += 1
              break
        count_before += 1
        temp -= 1
        i -= 1

    return count_before

def forecast_uni_with_gap(df, lag_value, steps_value, freq, gap_length, interval_length_before_gap, forecast_method="without_refit"):
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
        count_before = compute_count_before(df, freq, interval_length_before_gap)
        
        print(f"count_before: {count_before}")
        
        # Generate index for the predicted values
        last_index = df.index[-1]
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
        forecast_df = pd.DataFrame(data = pred.values, index = generated_indices, columns=["target"])
        return forecast_df

    else:
        temp_df = df.copy(deep = True)
        #last datetime  index
        last_index = df.index[-1]

        #reset the index of the temp_df to integer. 
        temp_df = temp_df.reset_index()
        temp_df = temp_df.drop(temp_df.columns[0], axis = 1)

        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, lags=lag_value, transformer_y=StandardScaler()
        )             
        
        pred_values = []
        
        for i in range(steps_value):
            forecaster.fit(temp_df.iloc[:,-1])
            
            pred_i = pd.DataFrame(forecaster.predict(steps = 1))
            pred_i_value = pred_i.iloc[0,0]
            pred_values.append(pred_i_value)

            #add the value to the temp_df
            temp_df.loc[len(temp_df)] = pred_i_value

        #=========================================================================    
        count_before = compute_count_before(df, freq, interval_length_before_gap)
        
        print(f"count_before: {count_before}")
        
        # Generate index for the predicted values
        last_index = df.index[-1]
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

        forecast_df = pd.DataFrame(data = pred_values, index = generated_indices, columns=["target"])
        return forecast_df


