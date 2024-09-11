import pandas as pd
import numpy as np


def compute_sma(df_arg, ts_type="univariate", window_sizes=[5, 10, 20]):
    df = df_arg.copy(deep=True)
    sma_df = df_arg.copy(deep=True)

    for window_size in window_sizes:
        if ts_type == "univariate":
            colname = df.columns[0]
            sma = df.iloc[:, -1].rolling(window=window_size).mean()
            sma_df[f"{colname}_sma_{window_size}"] = sma

        else:  # Multivariate case
            for i in range(len(df.columns)):
                sma = df.iloc[:, i].rolling(window=window_size).mean()
                colname = df.columns[i]
                sma_df[f"{colname}_sma_{window_size}"] = sma

    return sma_df
