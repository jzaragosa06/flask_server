import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response, Blueprint
from flask_cors import CORS
import io
import json


from app.resources.utility.lags import *

from app.resources.utility.gap_functions import *
from app.resources.utility.prepare_response import *

from app.resources.seasonality_analysis.seasonal import *
from app.resources.llm.gemini_pro_text import *
from app.resources.trend_prophet.trend import *
from app.resources.evaluation_forecast.forecast_uni import *
from app.resources.evaluation_forecast.forecast_multi import *

from app.resources.utility.dates import *

api = Blueprint("api", __name__)


@api.route("/forecast-univariate", methods=["POST"])
def forecast_univariate():
    file = request.files["inputFile"]
    if file:
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            print(df.head())
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            return jsonify({"message": f"Error loading DataFrame: {e}"}), 500

        tsType = request.form.get("type")
        freq = request.form.get("freq")
        description = request.form.get("description")
        steps = request.form.get("steps")
        forecastMethod = request.form.get("method")

        if freq not in ["D", "M", "Q", "Y"]:
            return jsonify({"message": "Format of the index is unsupported"}), 500

        # hasGap = checkGap(df=df, freq=freq)
        hasGap = False
        # we'll use a default of 7. This will be overwritten
        lag = 7

        exog = create_time_features(df=df, freq=freq)

        if hasGap != True:
            if forecastMethod == "without_refit":
                # from here, extract the metric, pred_test, pred_out
                result = evaluate_model_then_forecast_univariate(
                    df_arg=df,
                    exog=exog,
                    lag_value=lag,
                    freq=freq,
                    steps_value=int(steps),
                    forecast_method="without_refit",
                )

        else:
            gap_length, interval_length = identify_gap(df=df, freq=freq)

            if forecastMethod == "without_refit":
                ...
        try:
            response = prepare_forecast_response_univariate(
                df_arg=df,
                tsType=tsType,
                freq=freq,
                description=description,
                steps=int(steps),
                forecastMethod=forecastMethod,
                metric=result["mape"],
                pred_test=result["pred_test"],
                pred_out=result["pred_out"],
                result_dict=result,
            )

            return Response(json.dumps(response), mimetype="application/json")
        except Exception as e:
            print(f"Error in preparing response: {e}")
            return jsonify({"message": f"Error in preparing response: {e}"}), 500
    else:
        return jsonify({"message": "No csv file included"}), 500


@api.route("/forecast-multivariate", methods=["POST"])
def forecast_multivariate():
    file = request.files["inputFile"]

    if file:
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            print(df.head())
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            return jsonify({"message": f"Error loading DataFrame: {e}"}), 500

        tsType = request.form.get("type")
        freq = request.form.get("freq")
        description = request.form.get("description")
        steps = request.form.get("steps")
        forecastMethod = request.form.get("method")

        if freq not in ["D", "M", "Q", "Y"]:
            return jsonify({"message": "Format of the index is unsupported"}), 500

        # dict_lags, lag_list = sig_lag(df, 50, ts_type="multivariate")
        # we'll use a default of 7
        lag = 7

        exog = create_time_features(df=df, freq=freq)

        # hasGap = checkGap(df=df, freq=freq)
        # print("has gap? " + str(hasGap))
        hasGap = False

        if hasGap != True:
            if forecastMethod == "without_refit":
                result = evaluate_model_then_forecast_multivariate(
                    df_arg=df,
                    exog=exog,
                    lag_value=lag,
                    steps_value=int(steps),
                    freq=freq,
                    forecast_method="without_refit",
                )
        else:
            if forecastMethod == "without_refit":
                ...
        try:
            response = prepare_forecast_response_multivariate(
                df_arg=df,
                tsType=tsType,
                freq=freq,
                description=description,
                steps=steps,
                forecastMethod=forecastMethod,
                metric=result["mape"],
                pred_test=result["pred_test"],
                pred_out=result["pred_out"],
                result_dict=result,
            )

            print(response)
            return Response(json.dumps(response), mimetype="application/json")
        except Exception as e:
            print(f"Error in preparing response: {e}")
            return jsonify({"message": f"Error in preparing response: {e}"}), 500
    else:
        return jsonify({"message": "No csv file included"}), 500


@api.route("/trend", methods=["POST"])
def trend():
    file = request.files["inputFile"]

    if file:
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            return jsonify({"message": f"Error loading DataFrame: {e}"}), 500

        tsType = request.form.get("type")
        freq = request.form.get("freq")
        description = request.form.get("description")

        date_column = df.index.name
        colnames = df.columns.to_list()

        trend_result = compute_trend_prophet(
            df_arg=df, date_column=date_column, value_columns=colnames, freq=freq
        )

        if tsType == "univariate":
            response = prepare_trend_response_univariate(
                df_arg=df,
                tsType=tsType,
                trend_result=trend_result,
                freq=freq,
                description=description,
            )
        else:
            response = prepare_trend_response_multivariate(
                df_arg=df,
                tsType=tsType,
                trend_result=trend_result,
                freq=freq,
                description=description,
            )

        print(response)
        return Response(json.dumps(response), mimetype="application/json")
    else:
        return jsonify({"message": "No csv file included"}), 500


@api.route("/seasonality", methods=["POST"])
def seasonality():
    file = request.files["inputFile"]

    if file:
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            print(df.head())
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            return jsonify({"message": f"Error loading DataFrame: {e}"}), 500

        tsType = request.form.get("type")
        freq = request.form.get("freq")
        description = request.form.get("description")

        if freq not in ["D", "M", "Q", "Y"]:
            return jsonify({"message": "Format of the index is unsupported"}), 500

        date_column = df.index.name
        colnames = df.columns.to_list()

        components, seasonality_per_period = compute_seasonality_prophet(
            df_arg=df, date_column=date_column, value_columns=colnames, freq=freq
        )

        if tsType == "univariate":
            response = prepare_seasonality_response_univariate(
                df_arg=df,
                tsType=tsType,
                colnames=colnames,
                components=components,
                seasonality_per_period=seasonality_per_period,
                freq=freq,
                description=description,
            )
        else:
            response = prepare_seasonality_response_multivariate(
                df_arg=df,
                tsType=tsType,
                colnames=colnames,
                components=components,
                seasonality_per_period=seasonality_per_period,
                freq=freq,
                description=description,
            )

        return Response(json.dumps(response), mimetype="application/json")
    else:
        return jsonify({"message": "No csv file included"}), 500


@api.route("/llm", methods=["POST"])
def llm():
    data = request.json  # This will correctly get the JSON payload
    message = data.get("message")
    about = data.get("about")

    print(message)
    print(f"about {about}")

    # this returns a raw text
    response = answerMessage(message=message, about=about)

    print(response)

    return jsonify({"response": response})
