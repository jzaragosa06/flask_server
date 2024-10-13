# import pathlib
# import textwrap

# import google.generativeai as genai
# import markdown2


# model = genai.GenerativeModel("gemini-pro")

# api_key = "AIzaSyDUFnIcM040z-zIN-d5EL4FGzOj_Ps5ybs"
# genai.configure(api_key=api_key)


# def explainForecastBehavior(behaviorRaw):
#     response = model.generate_content(
#         f"This is a forecast. Explain the behavior of the time series forecast: {behaviorRaw}. Explain this in one paragraph. use future tense, since this is a forecast. describe this as a report."
#     )

#     # print(to_markdown(response.text))
#     # print(response.text)
#     return response.text


# def explainTrendBehavior(behaviorRaw):
#     response = model.generate_content(
#         f"This is a trend. Explain the behavior of the trend of time series: {behaviorRaw}. Explain this in one paragraph. use future tense. describe this as a report."
#     )

#     # print(to_markdown(response.text))
#     # print(response.text)
#     return response.text


# def explainSeasonalityBehavior(behaviorRaw):
#     response = model.generate_content(
#         f"This is a seasonality. Explain the behavior of the seasonality of time series: {behaviorRaw}. Explain this in one paragraph"
#     )

#     # print(to_markdown(response.text))
#     # print(response.text)
#     return response.text


# #     return response.text
# def answerMessage(message, about, text_result="None"):
#     # we can extract the 'about' i.e., trend, seasonality, forecast from the screen.
#     query = f"Answer this question: {message}."
#     response = model.generate_content(query)
#     text = markdown2.markdown(response.text)

#     return text


# def describeForecastModelOnTest(mape, about):
#     query = f"This is about {about}. In testing the model on testing data, i get a MAPE of {mape}. Describe how the model do. Explain it simply in one paragraph"
#     response = model.generate_content(query)
#     return response.text


import pathlib
import textwrap

import google.generativeai as genai
import markdown2


model = genai.GenerativeModel("gemini-pro")

api_key = "AIzaSyDUFnIcM040z-zIN-d5EL4FGzOj_Ps5ybs"
genai.configure(api_key=api_key)


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


# return response.text
def answerMessage(message, about, text_result="None"):
    # we can extract the 'about' i.e., trend, seasonality, forecast from the screen.
    query = f"Answer this question: {message}."
    response = model.generate_content(query)
    text = markdown2.markdown(response.text)

    return text


def describeForecastModelOnTest(mape, about):
    query = f"This is about {about}. In testing the model on testing data, i get a MAPE of {mape}. Describe how the model do. Explain it simply in one paragraph"
    response = model.generate_content(query)
    return response.text
