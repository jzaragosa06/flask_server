import pandas as pd
import numpy as np


def compute_sma(df_arg, ts_type="univariate"):
    if ts_type == "univariate":
        df = df_arg.copy()

        window_size = 10
        sma = df.iloc[:, -1].rolling(window=window_size).mean()
        sma_df = pd.DataFrame(data=sma.values, index=sma.index, columns=["target"])
        # how are we going to handle the NaN, generated by the window

        return sma_df
    else:
        df = df_arg.copy()
        window_size=10

        sma_df = pd.DataFrame()
        for i in range(len(df.columns)):
            if i != (len(df.columns) - 1): 
                sma = df.iloc[:, i].rolling(window = window_size).mean()
                sma_df[f"Feature_{i}"] = pd.DataFrame(sma)
            else:
                sma = df.iloc[:, i].rolling(window=window_size).mean()
                sma_df["target"] = pd.DataFrame(sma)
                
        return sma_df