# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt
# from datetime import datetime
# from tabulate import tabulate
# import json
# from pprint import pprint


# def seasonality_plot_df(m, ds):
#     """Prepare dataframe for plotting seasonal components.

#     Parameters
#     ----------
#     m: Prophet model.
#     ds: List of dates for column ds.

#     Returns
#     -------
#     A dataframe with seasonal components on ds.
#     """
#     df_dict = {"ds": ds, "cap": 1.0, "floor": 0.0}
#     for name in m.extra_regressors:
#         df_dict[name] = 0.0
#     # Activate all conditional seasonality columns
#     for props in m.seasonalities.values():
#         if props["condition_name"] is not None:
#             df_dict[props["condition_name"]] = True
#     df = pd.DataFrame(df_dict)
#     df = m.setup_dataframe(df)
#     return df


# def compute_seasonality_prophet(df_arg, date_column, value_columns, freq):
#     """
#     Compute seasonality for univariate or multivariate data using Prophet.

#     Parameters:
#     df_arg (pd.DataFrame): DataFrame containing the time series data.
#     date_column (str): Name of the column with dates.
#     value_columns (list of str): List of column names with time series values.

#     Returns:
#     dict: A dictionary with seasonality DataFrames for each value column.
#     """

#     seasonality_per_period = {}
#     model = None
#     components = None

#     # we will be removing the date as the index into a column.
#     df_deep_copy = df_arg.copy(deep=True)
#     df_deep_copy = df_deep_copy.reset_index()

#     # Prepare the data for each value column
#     for value_column in value_columns:
#         df = df_deep_copy[[date_column, value_column]].copy()
#         df.columns = ["ds", "y"]  # Prophet requires 'ds' (date) and 'y' (value) columns
#         df["ds"] = pd.to_datetime(df["ds"])

#         if freq == "D":
#             # Initialize the Prophet model
#             model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
#             components = ["weekly", "yearly"]
#         elif freq == "W":
#             model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
#             components = ["yearly"]

#         elif freq == "M":
#             model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
#             components = ["yearly"]

#         elif freq == "Y":
#             model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
#             components = ["yearly"]

#         model.fit(df)

#         # =====================================================================================
#         for component in components:
#             temp_dict = {}
#             if component == "weekly":
#                 # Compute weekly seasonality for a Sun-Sat sequence of dates.
#                 days = pd.date_range(start="2017-01-01", periods=7) + pd.Timedelta(
#                     days=0
#                 )
#                 df_w = seasonality_plot_df(model, days)
#                 seas = model.predict_seasonal_components(df_w)

#                 print(f"weekly seasonality - {seas}")
#                 # prepare to store in dictionary
#                 temp_dict["weekly"] = seas[["weekly", "weekly_lower", "weekly_upper"]]

#             if component == "yearly":
#                 # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates. arbitrary year.
#                 days = pd.date_range(start="2017-01-01", periods=365) + pd.Timedelta(
#                     days=0
#                 )
#                 df_y = seasonality_plot_df(model, days)
#                 seas = model.predict_seasonal_components(df_y)
#                 # prepare to store in dictionary
#                 temp_dict["yearly"] = seas[["yearly", "yearly_lower", "yearly_upper"]]
#             # then add the dict to the seasonality_er_period
#             seasonality_per_period[value_column] = temp_dict

#         print(f"seasonality per period full - {seasonality_per_period}")
#         # =====================================================================================

#     return components, seasonality_per_period


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate
import json
from pprint import pprint


def seasonality_plot_df(m, ds):
    """Prepare dataframe for plotting seasonal components.

    Parameters
    ----------
    m: Prophet model.
    ds: List of dates for column ds.

    Returns
    -------
    A dataframe with seasonal components on ds.
    """
    df_dict = {"ds": ds, "cap": 1.0, "floor": 0.0}
    for name in m.extra_regressors:
        df_dict[name] = 0.0
    # Activate all conditional seasonality columns
    for props in m.seasonalities.values():
        if props["condition_name"] is not None:
            df_dict[props["condition_name"]] = True
    df = pd.DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df


def compute_seasonality_prophet(df_arg, date_column, value_columns, freq):
    """
    Compute seasonality for univariate or multivariate data using Prophet.

    Parameters:
    df_arg (pd.DataFrame): DataFrame containing the time series data.
    date_column (str): Name of the column with dates.
    value_columns (list of str): List of column names with time series values.

    Returns:
    dict: A dictionary with seasonality DataFrames for each value column.
    """

    seasonality_per_period = {}
    model = None
    components = None

    # we will be removing the date as the index into a column.
    df_deep_copy = df_arg.copy(deep=True)
    df_deep_copy = df_deep_copy.reset_index()

    # Prepare the data for each value column
    for value_column in value_columns:
        df = df_deep_copy[[date_column, value_column]].copy()
        df.columns = ["ds", "y"]  # Prophet requires 'ds' (date) and 'y' (value) columns
        df["ds"] = pd.to_datetime(df["ds"])

        if freq == "D":
            # Initialize the Prophet model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            components = ["weekly", "yearly"]
        elif freq == "W":
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
            components = ["yearly"]

        elif freq == "M":
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
            components = ["yearly"]

        elif freq == "Y":
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
            components = ["yearly"]

        model.fit(df)
        temp_dict = {}
        # =====================================================================================
        for component in components:
            if component == "weekly":
                # Compute weekly seasonality for a Sun-Sat sequence of dates.
                days = pd.date_range(start="2017-01-01", periods=7) + pd.Timedelta(
                    days=0
                )
                df_w = seasonality_plot_df(model, days)
                seas = model.predict_seasonal_components(df_w)

                print(f"weekly seasonality - {seas}")
                # prepare to store in dictionary
                temp_dict["weekly"] = seas[["weekly", "weekly_lower", "weekly_upper"]]

            if component == "yearly":
                # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates. arbitrary year.
                days = pd.date_range(start="2017-01-01", periods=365) + pd.Timedelta(
                    days=0
                )
                df_y = seasonality_plot_df(model, days)
                seas = model.predict_seasonal_components(df_y)
                # prepare to store in dictionary
                temp_dict["yearly"] = seas[["yearly", "yearly_lower", "yearly_upper"]]

        # then add the dict to the seasonality_er_period
        seasonality_per_period[value_column] = temp_dict

        print(f"seasonality per period full - {seasonality_per_period}")
        # =====================================================================================

    return components, seasonality_per_period
