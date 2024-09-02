import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import io


# def prepare_json_response(
#     df_arg,
#     tsType,
#     freq,
#     description,
#     window_size,
#     hasSeasonal,
#     steps,
#     forecastMethod,
#     trend_result,
#     seasonal_result,
#     metric,
#     pred_test,
#     pred_out,
# ):
#     df = df_arg.copy(deep=True)

#     test_size = 0.2
#     test_samples = int(test_size * len(df))
#     train_data, test_data = df.iloc[:-test_samples], df.iloc[-test_samples:]

#     if tsType == "univariate":
#         trend_dict = trend_result.to_dict(orient="list")
#         trend_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(trend_result.index).to_list()
#         ]

#         seasonality_dict = seasonal_result.to_dict(orient="list")
#         seasonality_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(seasonal_result.index).to_list()
#         ]

#         pred_out_dict = pred_out.to_dict(orient="list")
#         pred_out_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(pred_out.index).to_list()
#         ]

#         pred_test_dict = pred_test.to_dict(orient="list")
#         pred_test_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(pred_test.index).to_list()
#         ]

#         train_data_dict = train_data.to_dict(orient="list")
#         train_data_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(train_data.index).to_list()
#         ]

#         test_data_dict = test_data.to_dict(orient="list")
#         test_data_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(test_data.index).to_list()
#         ]

#         df_dict = df.to_dict(orient="list")
#         df_dict["index"] = [
#             date.strftime("%m/%d/%Y") for date in pd.to_datetime(df.index).to_list()
#         ]

#         response = {
#             "metadata": {
#                 "tstype": tsType,
#                 "freq": freq,
#                 "description": description,
#                 "window_size": window_size,
#                 "seasonal": hasSeasonal,
#                 "steps": steps,
#                 "forecast_method": forecastMethod,
#             },
#             "trend": trend_dict,
#             "seasonality": seasonality_dict,
#             "forecast": {
#                 "pred_out": pred_out_dict,
#                 "pred_test": pred_test_dict,
#                 "metric": metric,
#             },
#             "data": {
#                 "train_data": train_data_dict,
#                 "test_data": test_data_dict,
#                 "entire_data": df_dict,
#             },
#         }

#     else:
#         trend_dict = trend_result.to_dict(orient="list")
#         trend_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(trend_result.index).to_list()
#         ]

#         seasonality_dict = seasonal_result.to_dict(orient="list")
#         seasonality_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(seasonal_result.index).to_list()
#         ]

#         pred_out_dict = pred_out.to_dict(orient="list")
#         pred_out_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(pred_out.index).to_list()
#         ]

#         pred_test_dict = pred_test.to_dict(orient="list")
#         pred_test_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(pred_test.index).to_list()
#         ]

#         train_data_dict = train_data.to_dict(orient="list")
#         train_data_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(train_data.index).to_list()
#         ]

#         test_data_dict = test_data.to_dict(orient="list")
#         test_data_dict["index"] = [
#             date.strftime("%m/%d/%Y")
#             for date in pd.to_datetime(test_data.index).to_list()
#         ]

#         df_dict = df.to_dict(orient="list")
#         df_dict["index"] = [
#             date.strftime("%m/%d/%Y") for date in pd.to_datetime(df.index).to_list()
#         ]

#         response = {
#             "metadata": {
#                 "tstype": tsType,
#                 "freq": freq,
#                 "description": description,
#                 "window_size": window_size,
#                 "seasonal": hasSeasonal,
#                 "steps": steps,
#                 "forecast_method": forecastMethod,
#             },
#             "trend": trend_dict,
#             "seasonality": seasonality_dict,
#             "forecast": {
#                 "pred_out": pred_out_dict,
#                 "pred_test": pred_test_dict,
#                 "metric": metric,
#             },
#             "data": {
#                 "train_data": train_data_dict,
#                 "test_data": test_data_dict,
#                 "entire_data": df_dict,
#             },
#         }

#     return response


def fillMissing(df):
    return df.fillna("")


def prepare_json_response(
    df_arg,
    tsType,
    freq,
    description,
    window_size,
    hasSeasonal,
    steps,
    forecastMethod,
    trend_result,
    seasonal_result,
    metric,
    pred_test,
    pred_out,
):
    df = df_arg.copy(deep=True)

    test_size = 0.2
    test_samples = int(test_size * len(df))
    train_data, test_data = df.iloc[:-test_samples], df.iloc[-test_samples:]

    trend_dict = fillMissing(trend_result).to_dict(orient="list")
    trend_dict["index"] = [
        date.strftime("%m/%d/%Y")
        for date in pd.to_datetime(trend_result.index).to_list()
    ]

    seasonality_dict = fillMissing(seasonal_result).to_dict(orient="list")
    seasonality_dict["index"] = [
        date.strftime("%m/%d/%Y")
        for date in pd.to_datetime(seasonal_result.index).to_list()
    ]

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

    print(f"metric: {metric}")

    response = {
        "metadata": {
            "tstype": tsType,
            "freq": freq,
            "description": description,
            "window_size": window_size,
            "seasonal": hasSeasonal,
            "steps": steps,
            "forecast_method": forecastMethod,
        },
        "trend": trend_dict,
        "seasonality": seasonality_dict,
        "forecast": {
            "pred_out": pred_out_dict,
            "pred_test": pred_test_dict,
            # "metric": metric.values
        },
        "data": {
            "train_data": train_data_dict,
            "test_data": test_data_dict,
            "entire_data": df_dict,
        },
    }

    return response
