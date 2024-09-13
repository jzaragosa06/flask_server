
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime


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
    
    
    seasonality_results = {}
    model = None
    components = None
    
    
    # we will be removing the date as the index into a column. 
    df_deep_copy = df_arg.copy(deep = True)
    df_deep_copy = df_deep_copy.reset_index()
    
    # Prepare the data for each value column
    for value_column in value_columns:
        df = df_deep_copy[[date_column, value_column]].copy()
        df.columns = ["ds", "y"]  # Prophet requires 'ds' (date) and 'y' (value) columns
        df['ds'] = pd.to_datetime(df['ds'])


        if freq == 'D':
            # Initialize the Prophet model
            model = Prophet(
                yearly_seasonality=True, weekly_seasonality=True
            )  
            components = ["ds", "weekly", "yearly" ]
        elif freq == 'W':
            model = Prophet(
                yearly_seasonality=True, weekly_seasonality=False
            )
            components = ["ds",  "yearly" ]
                      
        elif freq == 'M':
            model = Prophet(
                yearly_seasonality=True, weekly_seasonality=False
            )
            components = ["ds", "yearly" ]
            
        elif freq == 'Y':
            model = Prophet(
                yearly_seasonality=True, weekly_seasonality=False
            )
            components = ["ds", "yearly" ]
            
        
        model.fit(df)

        # Make future dataframe for predictions (no additional future predictions required)
        future = model.make_future_dataframe(periods=0)

        # Get the forecast and decomposition components
        forecast = model.predict(future)

        # Plot the decomposition graph (trend, seasonality)
        model.plot_components(forecast)
        plt.suptitle(f"Seasonality Components for {value_column}")
        plt.show()


        # Extract the seasonality components that are present
        seasonality_df = forecast[components]
        seasonality_results[value_column] = seasonality_df
    
    return seasonality_results, components


 

# Example usage
df = pd.read_csv(r"test/data/candy_production.csv", index_col = 0, parse_dates = True)
# df = pd.read_csv(r"test/data/thames_water.csv", index_col = 0, parse_dates = [0])
# df = pd.read_csv(r"test/data/apple2.csv", parse_dates=True, index_col=0)
df.index = pd.to_datetime(df.index, errors='coerce')  # Convert index to datetime explicitly

date_column = df.index.name
value_columns = df.columns.to_list()


print(df)

# we're passing a dataframe where the dates are set to the index. We need to write the code to 
# take this into account. 
seasonal_dfs = compute_seasonality_prophet(
    df_arg = df, date_column=date_column, value_columns=value_columns, freq='M'
)

# Print results
for col, seasonality_df in seasonal_dfs.items():
    print(f"Seasonality for {col}:")
    print(seasonality_df)
