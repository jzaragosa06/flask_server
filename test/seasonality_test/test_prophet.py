

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


def compute_seasonality_prophet(df_arg, date_column, value_column):
    # Prepare the data for Prophet
    df = df_arg[[date_column, value_column]].copy()
    df.columns = ["ds", "y"]  # Prophet requires 'ds' (date) and 'y' (value) columns

    # Initialize the Prophet model
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, 
    )  # Disable weekly seasonality
    model.fit(df)

    # Make future dataframe for predictions (no additional future predictions required)
    future = model.make_future_dataframe(periods=0)

    # Get the forecast and decomposition components
    forecast = model.predict(future)

    # Plot the decomposition graph (trend, seasonality)
    model.plot_components(forecast)
    plt.show()

    # Check if 'yearly' and 'trend' are in the forecast DataFrame and return only existing components
    components = ["ds", "trend"]
    if "yearly" in forecast.columns:
        components.append("yearly")
    if "weekly" in forecast.columns:  # Safe check for weekly
        components.append("weekly")

    # Extract the seasonality components that are present
    seasonality_df = forecast[components]

    return seasonality_df


df = pd.read_csv("test/data/candy_production.csv")
seasonal_df = compute_seasonality_prophet(
    df, date_column="observation_date", value_column="IPG3113N"
)

print(df)

print(seasonal_df)