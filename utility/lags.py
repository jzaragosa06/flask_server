import pandas as pd
import numpy as np
import statsmodels.api as sm


def sig_lag(df, max_lag, ts_type="univariate"):
    if ts_type == "univariate":
        acf = sm.tsa.stattools.acf(df, nlags=max_lag)
        acf_sig_lags = np.where(np.abs(acf) > 2 / np.sqrt(len(df)))[0]

        acf_sig_lag = acf_sig_lags[len(acf_sig_lags) - 1]

        return acf_sig_lag
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

        return acf_lags_muli
