import pandas as pd
import numpy as np

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect  import ForecasterAutoregDirect
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

#this is relative import
# from seasonality_analysis import *
from models.stacking import *


def forcast_uni(df_arg, lag_value, steps_value,freq, forecast_method = 'without_refit'):
    df = df_arg.copy()
    #get the model
    stacking_regressor = build_staking_regressor()

    if forecast_method == 'without_refit':
        forecaster = ForecasterAutoreg(
            regressor=stacking_regressor, 
            lags = lag_value,
            transformer_y= StandardScaler()
        )

        #fit the model
        forecaster.fit(df.iloc[:, -1])

        #predict
        pred = forecaster.predict(steps=steps_value)

        #we need to handle the gaps
        #we will solve it later 

        #we need to generate index for the predicted values.
        last_index = df.index[-1]
        new_indices = []

        #we add 1 since the range is exclusive on the upperbound.
        for i in range(1, steps_value + 1):
            if freq == 'D':
                new_index = last_index + pd.DateOffset(days=i)
            elif freq == 'W':
                new_index = last_index + pd.DateOffset(weeks=i)
            elif freq == 'M':
                new_index = last_index + pd.DateOffset(months=i)
            elif freq == 'Q':
                new_index = last_index + pd.DateOffset(months=3*i)
            elif freq == 'Y':
                new_index = last_index + pd.DateOffset(years=i)
            else:
                raise ValueError(f"Frequency '{freq}' is not supported")
            
            new_indices.append(new_index)

        #converting the list to datetimeindex
        new_indices = pd.DatetimeIndex(new_indices)

        #then create a new dataframe of result
        forecast_df = pd.DataFrame(data=pred.values, index=new_indices, columns=['target'])



    



    