import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate


def compute_trend_prophet(df_arg, date_column, value_columns, freq):
    """
    Compute trend for univariate or multivariate data using Prophet.

    Parameters:
    df_arg (pd.DataFrame): DataFrame containing the time series data.
    date_column (str): Name of the column with dates.
    value_columns (list of str): List of column names with time series values.

    Returns:
    dict: A dictionary with seasonality DataFrames for each value column.
    """

    trend_results = {}

    # we will be removing the date as the index into a column.
    df_deep_copy = df_arg.copy(deep=True)
    df_deep_copy = df_deep_copy.reset_index()

    # Prepare the data for each value column
    for value_column in value_columns:
        df = df_deep_copy[[date_column, value_column]].copy()
        df.columns = ["ds", "y"]  # Prophet requires 'ds' (date) and 'y' (value) columns
        df["ds"] = pd.to_datetime(df["ds"])

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        components = ["ds", "trend"]
        model.fit(df)

        # Make future dataframe for predictions (no additional future predictions required)
        future = model.make_future_dataframe(periods=0)

        # Get the forecast and decomposition components
        forecast = model.predict(future)

        # storing the dataframes (including the date and the treend) into a dictionary of dataframe, for each variable.
        trend_results[value_column] = forecast[components]

        # # Plot the decomposition graph (trend, seasonality)
        # model.plot_components(forecast)
        # plt.suptitle(f"Seasonality Components for {value_column}")
        # plt.show()

    #     # Extract the seasonality components that are present
    #     seasonality_df = forecast[components]
    #     seasonality_results[value_column] = seasonality_df

    #     # =====================================================================================
    #     # Compute and extract specific seasonal components like yearly or weekly
    #     for component in components:
    #         if component == "yearly":
    #             # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    #             days = pd.date_range(start='2017-01-01', periods=365)
    #             df_y = seasonality_plot_df(model, days)
    #             seas = model.predict_seasonal_components(df_y)
    #             seas_returned_values[value_column] = seas

    #     # =====================================================================================

    # return seasonality_results, seas_returned_values
    return trend_results


# # Example usage
# # df = pd.read_csv(r"test/data/candy_production.csv", index_col=0, parse_dates=True)
# df = pd.read_csv(r"test/data/apple2.csv", index_col=0, parse_dates=True)

# df.index = pd.to_datetime(
#     df.index, errors="coerce"
# )  # Convert index to datetime explicitly

# date_column = df.index.name
# value_columns = df.columns.to_list()

# # Compute seasonality
# trend_results = compute_trend_prophet(
#     df_arg=df, date_column=date_column, value_columns=value_columns, freq="M"
# )


# print(trend_results)
