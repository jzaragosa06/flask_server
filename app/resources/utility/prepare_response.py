import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io

from app.resources.behavior.extract_behavior import *
from app.resources.llm.gemini_pro_text import *


def fillMissing(df):
    return df.fillna("")


def prepare_trend_response_univariate(
    df_arg,
    tsType,
    trend_result,
    freq,
    description,
):
    df = df_arg.copy(deep=True)
    trend_dict = {}
    col = df.columns[0]

    counter = True
    for varname, trend_df in trend_result.items():
        if counter is True:
            trend_dict["index"] = [
                date.strftime("%m/%d/%Y")
                for date in pd.to_datetime(trend_df["ds"]).to_list()
            ]

            trend_dict[varname] = trend_df["trend"].to_list()
        else:
            trend_dict[varname] = trend_df["trend"].to_list()

    response = {
        "metadata": {
            "tstype": "univariate",
            "freq": freq,
            "description": description,
            "forecast_method": "without_refit",
            "colname": col,
        },
        "trend": trend_dict,
        "explanations": "lorem ipsum dorem ...",
    }

    return response


def prepare_trend_response_multivariate(
    df_arg,
    tsType,
    trend_result,
    freq,
    description,
):
    df = df_arg.copy(deep=True)
    trend_dict = {}
    cols = df.columns.tolist()

    counter = True
    for varname, trend_df in trend_result.items():
        if counter is True:
            trend_dict["index"] = [
                date.strftime("%m/%d/%Y")
                for date in pd.to_datetime(trend_df["ds"]).to_list()
            ]

            trend_dict[varname] = trend_df["trend"].to_list()
        else:
            trend_dict[varname] = trend_df["trend"].to_list()

    explanations = {}
    for col in cols:
        explanations[col] = "lorem ipsum dolor"

    response = {
        "metadata": {
            "tstype": "multivariate",
            "freq": freq,
            "description": description,
            "forecast_method": "without_refit",
            "colname": cols,
        },
        "trend": trend_dict,
        "explanations": explanations,
        
    }

    return response

def prepare_seasonality_response_univariate(
    df_arg, tsType, colnames, components, seasonality_per_period, freq, description
):
    df = df_arg.copy(deep=True)
    seasonality_per_period_dict = {}

    # Process seasonality per period (seasonality_per_period)
    for varname, period_dict in seasonality_per_period.items():
        temp_dict_new = {}
        for period, period_df in period_dict.items():
            temp_dict_new[period] = {
                "values": period_df[period].to_list(),
            }
        seasonality_per_period_dict[varname] = temp_dict_new
        
    explanations = {}
    for component in components:
        explanations[component] = "lorem ipsum dolor"

    response = {
        "metadata": {
            "tstype": "univariate",
            "freq": freq,
            "description": description,
            "forecast_method": "without_refit",
            "colname": df.columns[0],
        },
        "components": components,
        "seasonality_per_period": seasonality_per_period_dict,
        "explanations": explanations,
        
    }

    return response

def prepare_seasonality_response_multivariate(
    df_arg, tsType, colnames, components, seasonality_per_period, freq, description
):
    df = df_arg.copy(deep=True)
    seasonality_per_period_dict = {}
    cols = df.columns.tolist()

    # Process seasonality per period (seasonality_per_period)
    for varname, period_dict in seasonality_per_period.items():
        temp_dict_new = {}
        for period, period_df in period_dict.items():
            temp_dict_new[period] = {
                "values": period_df[period].to_list(),
            }
        seasonality_per_period_dict[varname] = temp_dict_new
        
    explanations = {}
    for component in components: 
        temp_dict = {}
        for col in cols: 
            temp_dict[col] = f"lorem ipusum {col}"
        explanations[component] = temp_dict
        
        
    response = {
        "metadata": {
            "tstype": "multivariate",
            "freq": freq,
            "description": description,
            "forecast_method": "without_refit",
            "colname": cols,
        },
        "components": components,
        "seasonality_per_period": seasonality_per_period_dict,
        "explanations": explanations,
    }
    return response







def prepare_forecast_response_univariate(
    df_arg,
    tsType,
    freq,
    description,
    steps,
    forecastMethod,
    metric,
    pred_test,
    pred_out,
):
    df = df_arg.copy(deep=True)

    test_size = 0.2
    test_samples = int(test_size * len(df))
    train_data, test_data = df.iloc[:-test_samples], df.iloc[-test_samples:]

    pred_out_dict = fillMissing(pred_out).to_dict(orient="list")
    pred_out_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(pred_out.index).to_list()
    ]

    pred_test_dict = fillMissing(pred_test).to_dict(orient="list")
    pred_test_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(pred_test.index).to_list()
    ]

    train_data_dict = fillMissing(train_data).to_dict(orient="list")
    train_data_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(train_data.index).to_list()
    ]

    test_data_dict = fillMissing(test_data).to_dict(orient="list")
    test_data_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(test_data.index).to_list()
    ]

    df_dict = fillMissing(df).to_dict(orient="list")
    df_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(df.index).to_list()
    ]

    # ==========================================================
    # The above code snippet is written in Python and performs the following tasks:
    # here, we will extract the forecast explanation
    # here we will just use the temporary mape to generate a an explanation about the test on the model.
    behaviorRaw = detect_changes_in_series(time_series=pred_out)
    pred_out_explanation = explainForecastBehavior(behaviorRaw=behaviorRaw)
    # pred_test_explanation = describeForecastModelOnTest(mape, about="forecast")
    # ==========================================================

    response = {
        "metadata": {
            "tstype": tsType,
            "freq": freq,
            "description": description,
            "steps": steps,
            "forecast_method": forecastMethod,
            "colname": df.columns[0],
        },
        "forecast": {
            "pred_out": pred_out_dict,
            "pred_test": pred_test_dict,
            "pred_out_explanation": pred_out_explanation,
            "explanation2": "lorem ipsum dolor ...",
            "explanation3": "lorem ipsum dolor ...",
            "explanation4": "lorem ipsum dolor ...",
            "pred_test_explanation": "lorem ipsum dolor ...",
            "metrics": {
                "mae": 0,
                "mse": 0,
                "rmse": 0,
                "mape": 0,
            },
        },
        "data": {
            "train_data": train_data_dict,
            "test_data": test_data_dict,
            "entire_data": df_dict,
        },
    }

    print(response)

    return response


def prepare_forecast_response_multivariate(
    df_arg,
    tsType,
    freq,
    description,
    steps,
    forecastMethod,
    metric,
    pred_test,
    pred_out,
):
    df = df_arg.copy(deep=True)
    test_size = 0.2
    test_samples = int(test_size * len(df))
    train_data, test_data = df.iloc[:-test_samples], df.iloc[-test_samples:]

    pred_out_dict = fillMissing(pred_out).to_dict(orient="list")
    pred_out_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(pred_out.index).to_list()
    ]

    pred_test_dict = fillMissing(pred_test).to_dict(orient="list")
    pred_test_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(pred_test.index).to_list()
    ]

    train_data_dict = fillMissing(train_data).to_dict(orient="list")
    train_data_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(train_data.index).to_list()
    ]

    test_data_dict = fillMissing(test_data).to_dict(orient="list")
    test_data_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(test_data.index).to_list()
    ]

    df_dict = fillMissing(df).to_dict(orient="list")
    df_dict["index"] = [
        date.strftime("%m/%d/%Y") for date in pd.to_datetime(df.index).to_list()
    ]

    # ==========================================================
    # The above code snippet is written in Python and performs the following tasks:
    # here, we will extract the forecast explanation
    # here we will just use the temporary mape to generate a an explanation about the test on the model.
    behaviorRaw = detect_changes_in_series(time_series=pred_out)
    pred_out_explanation = explainForecastBehavior(behaviorRaw=behaviorRaw)
    # pred_test_explanation = describeForecastModelOnTest(mape, about="forecast")
    # ==========================================================
    response = {
        "metadata": {
            "tstype": tsType,
            "freq": freq,
            "description": description,
            "steps": steps,
            "forecast_method": forecastMethod,
            "colname": df.columns.tolist(),
        },
        "forecast": {
            "pred_out": pred_out_dict,
            "pred_test": pred_test_dict,
            "pred_out_explanation": pred_out_explanation,
            "pred_test_explanation": "lorem ipsum dolor ...",
            "metrics": {
                "mae": 0,
                "mse": 0,
                "rmse": 0,
                "mape": 0,
            },
        },
        "data": {
            "train_data": train_data_dict,
            "test_data": test_data_dict,
            "entire_data": df_dict,
        },
    }

    print(response)

    return response
