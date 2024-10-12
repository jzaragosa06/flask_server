import pathlib
import textwrap

import google.generativeai as genai
import markdown2


# from IPython.display import display
# from IPython.display import Markdown


# def to_markdown(text):
#     text = text.replace(":", " *")
#     return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

model = genai.GenerativeModel("gemini-pro")

api_key = "AIzaSyDUFnIcM040z-zIN-d5EL4FGzOj_Ps5ybs"
genai.configure(api_key=api_key)


def explainForecastBehavior(behaviorRaw):
    response = model.generate_content(
        f"This is a forecast. Explain the behavior of the time series forecast: {behaviorRaw}. Explain this in one paragraph. use future tense, since this is a forecast. describe this as a report."
    )

    # print(to_markdown(response.text))
    # print(response.text)
    return response.text


def explainTrendBehavior(behaviorRaw):
    response = model.generate_content(
        f"This is a trend. Explain the behavior of the trend of time series: {behaviorRaw}. Explain this in one paragraph. use future tense. describe this as a report."
    )

    # print(to_markdown(response.text))
    # print(response.text)
    return response.text


def explainSeasonalityBehavior(behaviorRaw):
    response = model.generate_content(
        f"This is a seasonality. Explain the behavior of the seasonality of time series: {behaviorRaw}. Explain this in one paragraph"
    )

    # print(to_markdown(response.text))
    # print(response.text)
    return response.text


# def answerMessage(question, about, text_result):
#     # we can extract the 'about' i.e., trend, seasonality, forecast from the screen.
#     query = f"this is a question about {about}. This is the behaviour of data: {text_result}. Answer this question: {question}."
#     response = model.generate_content(query)

#     return response.text


# def answerMessage(message, about, text_result="None"):
#     # we can extract the 'about' i.e., trend, seasonality, forecast from the screen.
#     query = f"this is a question about {about}. This is the behaviour of data: {text_result}. Answer this question: {message}."
#     print(query)
#     response = model.generate_content(query)


#     return response.text
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
