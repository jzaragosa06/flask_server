"""
Here are the updates to this code: 
    -If it is univariate, it will return a single integer value.
    -If it is multivariate, it will return a dictionary. 
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


def sig_lag(df, max_lag, ts_type="univariate"):
    """
    Calculate the significant lag for univariate or multivariate time series data.
    
    Parameters:
    - df: pd.DataFrame containing the time series data.
    - max_lag: int, the maximum number of lags to consider.
    - ts_type: str, "univariate" for single series or "multivariate" for multiple series.
    
    Returns:
    - If ts_type is "univariate", returns an int representing the significant lag.
    - If ts_type is "multivariate", returns a tuple containing:
      - dict_lags: dict with column names as keys and significant lags as values.
      - acf_lags_muli: list of significant lags for each column.
    """
    if ts_type == "univariate":
        acf = sm.tsa.stattools.acf(df, nlags=max_lag)
        acf_sig_lags = np.where(np.abs(acf) > 2 / np.sqrt(len(df)))[0]

        acf_sig_lag = acf_sig_lags[len(acf_sig_lags) - 1]

        return int(acf_sig_lag)
    else:
        # for each of the vriables, we'll find the significant lag.
        # we'll just align it to the order of the columns in df.

        acf_lags_muli = []
        for i in range(len(df.columns)):
            # we'll start with features and end with target.
            acf = sm.tsa.stattools.acf(df.iloc[:, i], nlags=max_lag)
            acf_sig_lags = np.where(np.abs(acf) > 2 / np.sqrt(len(df)))[0]

            acf_sig_lag = acf_sig_lags[len(acf_sig_lags) - 1]

            acf_lags_muli.append(acf_sig_lag)
            # acf_lags_muli.append({f"feature_{i}": acf_sig_lag})

        dict_lags = {}
        #dictionary of lags
        for i in range(len(df.columns)):
            dict_lags[df.columns[i]] = int(acf_lags_muli[i])

        #the first returns the dictionary format, and the second returns the list format. 
        return dict_lags, acf_lags_muli
