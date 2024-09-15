import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import io


from app.resources.behavior.extract_behavior import *
from app.resources.llm.gemini_pro_text import *


def fillMissing(df):
    return df.fillna("")


def prepare_trend_response(
    df_arg,
    tsType,
    trend_result,
):
    df = df_arg.copy(deep=True)

    if tsType == "univariate":
        colnames = df.columns[0]

        # here, we will extract the forecast explanation
        behaviorRaw = detect_changes_in_series(time_series=df)
        explanation = explainTrendBehavior(behaviorRaw=behaviorRaw)

    else:
        colnames = df.columns.tolist()

        explanation = {}

        for colname in colnames:
            temp_df = pd.DataFrame(df[colname])
            behaviorRaw = detect_changes_in_series(time_series=temp_df)
            explanation_text = explainTrendBehavior(behaviorRaw=behaviorRaw)
 
            explanation[colname] = explanation_text

    trend_dict = fillMissing(trend_result).to_dict(orient="list")
    
    trend_dict["index"] = [
        date.strftime("%m/%d/%Y")
        for date in pd.to_datetime(trend_result.index).to_list()
    ]

    response = {
        "tstype": tsType,
        "trend": trend_dict,
        "explanations": explanation,
        "colname": colnames,
    }

    return response


def prepare_seasonality_response(df_arg, tsType, colnames, components,  seasonal_dfs, seasonality_per_period):
    df = df_arg.copy(deep=True)
    #the seasonal_dfs is a dictionary of dataframe
    
    seasonality_dict = {}
    seasonality_per_period_dict = {}
    
    
    for varname, df in seasonal_dfs.items():
        temp_dict = {}
        for component in components: 
            
            if (component == "ds"):
                temp_dict[component] = [ date.strftime("%m/%d/%Y") for date in pd.to_datetime(df[component]).to_list()]
            else:     
                temp_dict[component] = df[component].to_list()
        
        seasonality_dict[varname] = temp_dict
        
    print(seasonality_dict)
    
    
    # Process seasonality per period (seasonality_per_period)
    for varname, period_dict in seasonality_per_period.items():
        
        
        for period, period_df in period_dict.items():
            temp_dict_new = {}
            temp_dict_new[period] =  {
                "values": period_df[period].to_list(),
                "lower": period_df[f"{period}_lower"].to_list(),
                "upper": period_df[f"{period}_upper"].to_list(),
            }
        
        #add to the seasonality_per_period_dict
        seasonality_per_period_dict[varname] = temp_dict_new
        
            
            
            
            
         


    response = {
        "tstype": tsType,
        "seasonality": seasonality_dict,
        "seasonality_per_period": seasonality_per_period_dict
    }

    return response


def prepare_forecast_response(
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

    # here, we will extract the forecast explanation
    behaviorRaw = detect_changes_in_series(time_series=pred_out)
    explanation = explainForecastBehavior(behaviorRaw=behaviorRaw)

    print(f"here is the explanation: {explanation}")

    # metric_type = type(metric)
    # print(f" the metric is : {metric}")
    # print(f" the metric is val : {metric.values}")
    # print(f" the metric is : {metric_type}")

    # Check the type of metric
    if isinstance(metric, pd.DataFrame):
        # Extract the single metric value for multivariate (assuming the metric is in the first row/column)
        metric_value = metric.iloc[
            0, 1
        ]  # Extracts the value from the first row and second column
    else:
        # Univariate case, just a float value
        metric_value = metric

    if tsType == "univariate":
        colnames = df.columns[0]
    else:
        colnames = df.columns.tolist()

    response = {
        "metadata": {
            "tstype": tsType,
            "freq": freq,
            "description": description,
            "steps": steps,
            "forecast_method": forecastMethod,
            "colname": colnames,
        },
        "forecast": {
            "pred_out": pred_out_dict,
            "pred_test": pred_test_dict,
            "pred_out_explanation": explanation,
            "metric": metric_value,
        },
        "data": {
            "train_data": train_data_dict,
            "test_data": test_data_dict,
            "entire_data": df_dict,
        },
    }

    return response
