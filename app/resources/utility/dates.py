from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def generate_list_of_dates():
    # Start date (January 1st)
    start_date = datetime(2024, 1, 1)

    # End date (December 31st)
    end_date = datetime(2024, 12, 31)

    # List to hold the formatted date strings
    date_list = []

    # Iterate through the range of dates
    current_date = start_date
    while current_date <= end_date:
        # Format the date without the year and add to list (e.g., "Jan. 24")
        date_str = current_date.strftime("%b %d")
        date_list.append(date_str)

        # Move to the next day
        current_date += timedelta(days=1)

    return date_list


def create_time_features(df, freq="D"):
    """
    Function to create time-based features based on the frequency of the data.

    Parameters:
    - df(pd.DataFrame): DataFrame with a DateTime index.
    - freq (str): Frequency of the data ('D', 'W', 'M', 'Q', 'Y').
                  'D' = Daily
                  'W' = Weekly
                  'M' = Monthly
                  'Q' = Quarterly
                  'Y' = Yearly

    Returns:
    - exog (pd.DataFrame): DataFrame with added time-based features.
    """

    # Ensure the index is a DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DateTimeIndex.")

    exog = pd.DataFrame()
    # Time-based features applicable for all frequencies
    exog["year"] = df.index.year
    exog["quarter"] = df.index.quarter

    if freq == "D":
        # Day-level features
        exog["day_of_week"] = df.index.dayofweek  # 0 = Monday, 6 = Sunday
        exog["day_of_month"] = df.index.day
        exog["day_of_year"] = df.index.dayofyear
        # exog['week_of_year'] = df.index.isocalendar().week

    elif freq == "W":
        # Week-level features
        exog["week_of_year"] = df.index.isocalendar().week
        exog["day_of_week"] = df.index.dayofweek

    elif freq == "M":
        # Month-level features
        exog["month"] = df.index.month
        exog["day_of_month"] = df.index.day

    elif freq == "Q":
        # Quarter-level features
        exog["quarter"] = df.index.quarter

    elif freq == "Y":
        # Year-level features
        exog["year"] = df.index.year

    else:
        raise ValueError("Unsupported frequency. Choose from 'D', 'W', 'M', 'Q', 'Y'.")

    return exog


def infer_frequency(df):
    # Get the index of the dataframe
    index = df.index

    # Calculate the difference between consecutive dates
    diff = index.to_series().diff().dropna()

    # Find the most common difference (mode)
    mode_diff = diff.mode()[0]

    # Determine the frequency based on the difference
    if pd.Timedelta(days=28) <= mode_diff <= pd.Timedelta(days=31):
        return "M"  # Monthly
    elif pd.Timedelta(days=89) <= mode_diff <= pd.Timedelta(days=92):
        return "Q"  # Quarterly
    elif pd.Timedelta(days=364) <= mode_diff <= pd.Timedelta(days=366):
        return "Y"  # Yearly
    elif mode_diff == pd.Timedelta(days=7):
        return "W"  # Weekly
    elif mode_diff == pd.Timedelta(days=1):
        return "D"  # Daily
    else:
        return "Unknown frequency"
