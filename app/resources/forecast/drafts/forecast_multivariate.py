
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

from models.stacking import *

def forecast_multi(df, lag_list, steps_value, freq, forecast_method='without_refit'):
    dict_lags = {}
    #dictionary of lags
    for i in range(len(df.columns)):
        dict_lags[df.columns[i]] = lag_list[i]

    last_col = df.columns[-1]
    
    stacking_regressor = build_staking_regressor()

    if forecast_method == 'without_refit':
        forecaster = ForecasterAutoregMultiVariate(
            regressor=stacking_regressor, 
            level=last_col, 
            lags=dict_lags, 
            # lags=lag_list,
            steps=steps_value, 
            transformer_series=StandardScaler(),
            transformer_exog=None, 
            weight_func=None, 
        )

        forecaster.fit(df)
        
        # Predict
        pred = forecaster.predict(steps=steps_value)

        # Generate index for the predicted values
        last_index = df.index[-1]
        new_indices = pd.date_range(start=last_index, periods=steps_value + 1, freq=freq)[1:]

        # Create a new DataFrame of the result
        forecast_df = pd.DataFrame(data=pred.values, index=new_indices, columns=['target'])

        return forecast_df



    



