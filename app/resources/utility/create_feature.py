import pandas as pd
import numpy as np
import copy


def create_lag_features_uni(df_arg, lag_value):
    #create a deep copy of the dataframe
    lagged_data = copy.deepcopy(df_arg)

    for i in range (1, lag_value + 1):
        lagged_data[f"lf{i}"] = df_arg.shift(i)
    
    # return lagged_data.ffill()
    return lagged_data.dropna()

#the lags must be in dictionary format. 
#take into account the ordering
def create_lag_features_multi(df_arg, dict_lags):
    lagged_data = copy.deepcopy(df_arg)
    for series_name, lag in dict_lags.items():
        for i in range(1, lag + 1):
            lagged_data[f"{series_name}_lf{i}"] = df_arg[series_name].shift(i)
    
    return lagged_data.dropna()
        

