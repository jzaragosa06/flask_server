import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


def compute_seasonality(df_arg, ts_type="univariate"):
    # we need to extract the period of the variable
    if ts_type == "univariate":
        df = df_arg.copy()
        period = 12  # we need to compute for this

        # we'll just omit the period for now
        stl = STL(df.iloc[:, -1])
        result = stl.fit()

        seasonal_df = pd.DataFrame(
            data=result.seasonal.values, index=result.seasonal.index, columns=["target"]
        )
        return seasonal_df
    else:
        df = df_arg.copy()
        period = []
        seasonal_df

        for i in range(len(df.columns)):
            if i != (len(df.columns) - 1):
                stl = STL(df.iloc[:, i])
                result = stl.ft()

                seasonal_df[f"feature_{i}"] = pd.DataFrame(result.seasonal)

        return seasonal_df
