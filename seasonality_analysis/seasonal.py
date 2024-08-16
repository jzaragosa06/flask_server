import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


# def compute_seasonality(df_arg, ts_type="univariate"):
#     # we need to extract the period of the variable
#     if ts_type == "univariate":
#         df = df_arg.copy(deep = True)
#         period = 12  # we need to compute for this

#         # we'll just omit the period for now
#         stl = STL(df.iloc[:, -1])
#         result = stl.fit()

#         seasonal_df = pd.DataFrame(
#             data=result.seasonal.values, index=result.seasonal.index, columns=["target"]
#         )
#         return seasonal_df
#     else:
#         df = df_arg.copy(deep = True)
#         period = []
#         seasonal_df

#         for i in range(len(df.columns)):
#             if i != (len(df.columns) - 1):
#                 stl = STL(df.iloc[:, i])
#                 result = stl.ft()

#                 seasonal_df[f"feature_{i}"] = pd.DataFrame(result.seasonal)

#         return seasonal_df


def compute_seasonality(df_arg, ts_type="univariate"):
    # we need to extract the period of the variable
    if ts_type == "univariate":
        df = df_arg.copy(deep=True)
        period = 12  # we need to compute for this

        try:
            stl = STL(df.iloc[:, -1])
            result = stl.fit()
            seasonal_df = pd.DataFrame(
                data=result.seasonal.values,
                index=result.seasonal.index,
                columns=["target"],
            )
            print(type(seasonal_df))
            return seasonal_df
        except Exception as e:
            print(f"Error in STL decomposition: {e}")
            return None  # or handle it in another appropriate way

    else:
        df = df_arg.copy(deep=True)
        period = []
        seasonal_df = pd.DataFrame()

        for i in range(len(df.columns)):
            if i != (len(df.columns) - 1):
                try:
                    stl = STL(df.iloc[:, i])
                    result = stl.fit()
                    seasonal_df[f"feature_{i}"] = pd.DataFrame(result.seasonal)
                except Exception as e:
                    print(f"Error in STL decomposition for column {i}: {e}")
                    seasonal_df[f"feature_{i}"] = pd.Series(
                        [np.nan] * len(df), index=df.index
                    )

        print(type(seasonal_df))
        return seasonal_df
