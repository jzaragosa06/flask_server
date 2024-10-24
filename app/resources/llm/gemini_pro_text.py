import pathlib
import textwrap
import google.generativeai as genai
import markdown2
from app.resources.utility.dates import generate_list_of_dates

model = genai.GenerativeModel("gemini-pro")

api_key = "AIzaSyDUFnIcM040z-zIN-d5EL4FGzOj_Ps5ybs"
genai.configure(api_key=api_key)


# =================================================================================================
def describeOutForecast_univariate(forecast, col, description=""):
    dates = forecast.index.strftime("%m-%d-%Y").tolist()
    # we'll get the last.
    values = forecast.values.tolist()

    # The first query is for describing the forecast movement.
    query1 = f"This is about time series forecast. Describe the forecast for {col}. here's the date: {dates} and here is the values: {values}. Particularly, describe the movement"
    # The second query is for describing how the variable usually moves. For this we will use the description.
    # We will require that in the description, the user describe the data/variables.
    query2 = f"This is about the time series forecast. The variable is {col}. The forecast values is {values}, where dates is {dates}. Describe how the time series of this variable usually moves."
    # The third query is to look for the possible factors that can affect the time series of variable.
    query3 = f"This is about the time series forecast. The variable is {col}. The forecast values is {values}, where dates is {dates}. Explore factors that affects the forecast."

    response1 = model.generate_content(query1)
    response2 = model.generate_content(query2)
    response3 = model.generate_content(query3)

    return {
        "response1": markdown2.markdown(response1.text),
        "response2": markdown2.markdown(response2.text),
        "response3": markdown2.markdown(response3.text),
    }


def describeOutForecast_multivariate(forecast, cols, description=""):
    dates = forecast.index.strftime("%m-%d-%Y").tolist()
    # we'll get the last. since it is the target. on the second thought, we're just returning the a single column in forecast.
    values = forecast.values.tolist()

    # The first query is for describing the forecast movement.
    query1 = f"This is about time series forecast. Describe the forecast for {cols[-1]}. here's the date: {dates} and here is the values: {values}. Particularly, describe the movement"
    # The second query is for describing how the variable usually moves. For this we will use the description.
    # We will require that in the description, the user describe the data/variables.
    query2 = f"This is about the time series forecast. The variable is {cols[-1]}. The forecast values is {values}, where dates is {dates}. Describe how the time series of this variable usually moves."
    # The third query looks for how the target is affected by other variables used in the forecast.
    query3 = f"This is about the time series forecast. The target variable is {cols[-1]}. The forecast values is {values}, where dates is {dates}. describe these variables used in forecast affected the target: {cols}"
    # The fourth query is to look for the possible factors that can affect the time series of variable.
    query4 = f"This is about the time series forecast. The target variable is {cols[-1]}. The forecast values is {values}, where dates is {dates}. Explore factors that affects the forecast besides {cols}"

    response1 = model.generate_content(query1)
    response2 = model.generate_content(query2)
    response3 = model.generate_content(query3)
    response4 = model.generate_content(query4)

    return {
        "response1": markdown2.markdown(response1.text),
        "response2": markdown2.markdown(response2.text),
        "response3": markdown2.markdown(response3.text),
        "response4": markdown2.markdown(response4.text),
    }


def describeTestForecast(forecast, cols, metrics, error, description=""):
    error_list = error.values.tolist()
    # this query describe the performance of the model based on the metrics.
    query1 = f"This is about the performance of the model in time series forecast. Here is the error (difference between the predicted and the original): {error_list} Given the Mean Absolute Percentage Error (MAPE) on test set is {metrics} (i.e., convert to % by multiplying 100), describe the performance of the model. Describe whether using it on out-sample forecast is good."

    response1 = model.generate_content(query1)

    return {
        "response1": markdown2.markdown(response1.text),
    }


# =================================================================================================
def generate_explanations_univariate(period, col, values, days, description):
    """Generate explanations based on the time series data and period."""
    query1 = f"I've extracted the {period} seasonality component of {col} time series data. These are the values: {values} and these are the dates: {days}. Describe the result, identify when the values are above and below average, and explain what it means. Be comprehensive."
    query2 = f"I've extracted the {period} seasonality component of {col} time series data. These are the values: {values} and these are the dates: {days}. Explore factors/variables that affect the results. Be comprehensive."

    response1 = model.generate_content(query1)
    response2 = model.generate_content(query2)

    return {
        "response1": markdown2.markdown(response1.text),
        "response2": markdown2.markdown(response2.text),
    }


def generate_explanations_multivariate(period, col, values, days, description):
    """Generate explanations based on the time series data and period."""
    query1 = f"I've extracted the {period} seasonality component of {col} time series data. These are the values: {values} and these are the dates: {days}. Describe the result, identify when the values are above and below average, and explain what it means. Be comprehensive. Next explore how other factors/variables affect the results. "
    # query2 = f"I've extracted the {period} seasonality component of {col} time series data. These are the values: {values} and these are the dates: {days}. Explore factors/variables that affect the results. Be comprehensive."

    response1 = model.generate_content(query1)
    # response2 = model.generate_content(query2)

    return {
        "response1": markdown2.markdown(response1.text),
        # "response2": markdown2.markdown(response2.text),
    }


def describe_seasonality_univariate(seasonality_per_period, description=""):
    """Describe the univariate seasonality of a time series."""
    explanations = {}

    for varname, period_dict in seasonality_per_period.items():
        for period, period_df in period_dict.items():
            values = period_df[period].to_list()

            if period == "weekly":
                days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                explanations[period] = generate_explanations_univariate(
                    period=period,
                    col=varname,
                    values=values,
                    days=days,
                    description=description,
                )

            elif period == "yearly":
                days = generate_list_of_dates()
                explanations[period] = generate_explanations_univariate(
                    period=period,
                    col=varname,
                    values=values,
                    days=days,
                    description=description,
                )

    return explanations


def describe_seasonality_multivariate(
    seasonality_per_period, components, description=""
):
    """Describe the multivariate seasonality of a time series for specific components."""
    explanations = {}
    days_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days_year = generate_list_of_dates()

    # Initialize temporary dictionaries for each component
    temp_weekly = {}
    temp_yearly = {}

    for varname, period_dict in seasonality_per_period.items():
        for period, period_df in period_dict.items():
            values = period_df[period].to_list()

            if period == "weekly" and period in components:
                temp_weekly[varname] = generate_explanations_multivariate(
                    period=period,
                    col=varname,
                    values=values,
                    days=days_week,
                    description=description,
                )

            elif period == "yearly" and period in components:
                temp_yearly[varname] = generate_explanations_multivariate(
                    period=period,
                    col=varname,
                    values=values,
                    days=days_year,
                    description=description,
                )

    # Populate the final explanations dictionary based on components
    if "weekly" in components:
        explanations["weekly"] = temp_weekly
    if "yearly" in components:
        explanations["yearly"] = temp_yearly

    return explanations


# =================================================================================================
def describe_trend_univariate(trend_result, description=""):
    col = None
    for varname, trend_df in trend_result.items():
        # Keep every 5th row. This is to convserve token
        trend_df = trend_df.iloc[::5]

        values = trend_df["trend"].tolist()
        # we're converting the ds column to list. we use astype since each element is not datetimeindex obj.
        dates = trend_df["ds"].astype(str).tolist()
        col = varname

    # The first query is for describing the movement
    query1 = f"I've extracted the trend of {col} time series. here's the date: {dates} and here is the values: {values} (note, both list contains every 5th row from the original dataset, reducing the number of rows while retaining pattern. Do not mention this on result). Particularly, describe the movement of the trend"
    query2 = f"Given a time series variable {col}.Describe how the trend of this time series variable usually moves."
    # The third query is to look for the possible factors that can affect the time series of variable.
    query3 = f"Given a time series variable {col},  explore factors that affects the trend of this variable."

    response1 = model.generate_content(query1)
    response2 = model.generate_content(query2)
    response3 = model.generate_content(query3)

    return {
        "response1": markdown2.markdown(response1.text),
        "response2": markdown2.markdown(response2.text),
        "response3": markdown2.markdown(response3.text),
    }

def describe_trend_multivariate(trend_result, cols, description=""):
    explanations = {}

    for varname, trend_df in trend_result.items():
        # Keep every 8th row. This is to convserve token
        trend_df = trend_df.iloc[::8]

        values = trend_df["trend"].tolist()
        # we're converting the ds column to list. we use astype since each element is not datetimeindex obj.
        dates = trend_df["ds"].astype(str).tolist()

        # The first query is for describing the movement
        query1 = f"I've extracted the trend of {varname} time series. here's the date: {dates} and here is the values: {values} (note, both list contains every 8th row from the original dataset, reducing the number of rows while retaining pattern. Do not mention this on result). Particularly, describe the movement of the trend. Furthermore, describe how the trend of this time series variable usually move"
        # The third query is to look for the possible factors that can affect the time series of variable.
        query2 = f"Given a time series variable {varname},  explore how these time series variables: {cols} affect the {varname}"

        response1 = model.generate_content(query1)
        response2 = model.generate_content(query2)

        explanations[varname] = {
            "response1": markdown2.markdown(response1.text),
            "response2": markdown2.markdown(response2.text),
        }

    return explanations


# =================================================================================================


# return response.text
def answerMessage(message, about, text_result="None"):
    # we can extract the 'about' i.e., trend, seasonality, forecast from the screen.
    query = f"Answer this question: {message}."
    response = model.generate_content(query)
    text = markdown2.markdown(response.text)

    return text
