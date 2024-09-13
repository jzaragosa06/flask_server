import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response, Blueprint
from flask_cors import CORS
import io
import json



# from .resources.utility.lags import *
# from .resources.forecast.forecast_uni import *
# from .resources.forecast.forecast_uni_with_gap import *
# from .resources.forecast.forecast_multi import *
# from .resources.forecast.forecast_multi_with_gap import *
# from .resources.utility.gap_functions import *
# from .resources.Trend_analysis.sma import *
# from .resources.seasonality_analysis.seasonal import *
# from .resources.utility.prepare_response import *


from app.resources.utility.lags import *
from app.resources.forecast.forecast_uni import *
from app.resources.forecast.forecast_uni_with_gap import *
from app.resources.forecast.forecast_multi import *
from app.resources.forecast.forecast_multi_with_gap import *
from app.resources.utility.gap_functions import *
from app.resources.Trend_analysis.sma import *
from app.resources.seasonality_analysis.seasonal import *
from app.resources.utility.prepare_response import *



api = Blueprint('api', __name__)



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

        hasGap = checkGap(df=df, freq=freq)
        print("has gap? " + str(hasGap))

        lag = sig_lag(df, 50, ts_type="univariate")

        if hasGap != True:
            if forecastMethod == "without_refit":
                metric, pred_test = evaluate_model_uni(
                    df_arg=df,
                    lag_value=int(lag),
                    steps_value=int(steps),
                    freq=freq,
                    forecast_method="without_refit",
                )

                pred_out = forecast_uni(
                    df_arg=df,
                    lag_value=int(lag),
                    steps_value=int(steps),
                    freq=freq,
                    forecast_method="without_refit",
                )
            else:
                metric, pred_test = evaluate_model_uni(
                    df_arg=df,
                    lag_value=int(lag),
                    steps_value=int(steps),
                    freq=freq,
                    forecast_method="with_refit",
                )

                pred_out = forecast_uni(
                    df_arg=df,
                    lag_value=int(lag),
                    steps_value=int(steps),
                    freq=freq,
                    forecast_method="with_refit",
                )
        else:
            gap_length, interval_length = identify_gap(df=df, freq=freq)

            if forecastMethod == "without_refit":
                metric, pred_test = forecast_uni_with_gap(
                    df_arg=df,
                    lag_value=int(lag),
                    steps_value=int(steps),
                    freq=freq,
                    gap_length=gap_length,
                    interval_length_before_gap=interval_length,
                    forecast_method="without_refit",
                )

                pred_out = evaluate_model_uni_with_gap(
                    df_arg=df,
                    lag_value=int(lag),
                    steps_value=int(steps),
                    freq=freq,
                    gap_length=gap_length,
                    interval_length_before_gap=interval_length,
                    forecast_method="without_refit",
                )

            else:
                metric, pred_test = forecast_uni_with_gap(
                    df_arg=df,
                    lag_value=int(lag),
                    steps_value=int(steps),
                    freq=freq,
                    gap_length=gap_length,
                    interval_length_before_gap=interval_length,
                    forecast_method="with_refit",
                )

                pred_out = evaluate_model_uni_with_gap(
                    df_arg=df,
                    lag_value=int(lag),
                    steps_value=int(steps),
                    freq=freq,
                    gap_length=gap_length,
                    interval_length_before_gap=interval_length,
                    forecast_method="with_refit",
                )
        try:
            response = prepare_forecast_response(
                df_arg=df,
                tsType=tsType,
                freq=freq,
                description=description,
                steps=steps,
                forecastMethod=forecastMethod,
                metric=metric,
                pred_test=pred_test,
                pred_out=pred_out,
            )
            print(response)
            return Response(json.dumps(response), mimetype="application/json")
        except Exception as e:
            print(f"Error in preparing response: {e}")
            return jsonify({"message": f"Error in preparing response: {e}"}), 500


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

        dict_lags, lag_list = sig_lag(df, 50, ts_type="multivariate")

        hasGap = checkGap(df=df, freq=freq)
        print("has gap? " + str(hasGap))

        if hasGap != True:
            if forecastMethod == "without_refit":
                metric, pred_test = evaluate_model_multi(
                    df_arg=df,
                    dict_lags=dict_lags,
                    steps_value=int(steps),
                    freq=freq,
                    forecast_method="without_refit",
                )

                pred_out = forecast_multi(
                    df_arg=df,
                    dict_lags=dict_lags,
                    steps_value=int(steps),
                    freq=freq,
                    forecast_method="without_refit",
                )
            else:
                ...
        else:
            if forecastMethod == "without_refit":
                gap_length, interval_length = identify_gap(df=df, freq=freq)

                metric, pred_test = evaluate_model_multi_with_gap(
                    df_arg=df,
                    dict_lags=dict_lags,
                    steps_value=int(steps),
                    freq=freq,
                    gap_length=gap_length,
                    interval_length_before_gap=interval_length,
                    forecast_method="without_refit",
                )

                pred_out = forecast_multi_with_gap(
                    df_arg=df,
                    dict_lags=dict_lags,
                    steps_value=int(steps),
                    freq=freq,
                    gap_length=gap_length,
                    interval_length_before_gap=interval_length,
                    forecast_method="without_refit",
                )
            else:
                ...
        try:
            response = prepare_forecast_response(
                df_arg=df,
                tsType=tsType,
                freq=freq,
                description=description,
                steps=steps,
                forecastMethod=forecastMethod,
                metric=metric,
                pred_test=pred_test,
                pred_out=pred_out,
            )

            print(response)
            return Response(json.dumps(response), mimetype="application/json")
        except Exception as e:
            print(f"Error in preparing response: {e}")
            return jsonify({"message": f"Error in preparing response: {e}"}), 500


@api.route("/trend", methods=["POST"])
def trend():
    file = request.files["inputFile"]

    if file:
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            print(df.head())
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            return jsonify({"message": f"Error loading DataFrame: {e}"}), 500

        tsType = request.form.get("type")

        # extract trend and seasonality behaviour of the ts data.
        trend_result = compute_sma(df_arg=df, ts_type=tsType, window_sizes=[5, 10, 20])

        response = prepare_trend_response(
            df_arg=df, tsType=tsType, trend_result=trend_result
        )

        print(response)
        return Response(json.dumps(response), mimetype="application/json")


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

        
        date_column = df.index.name
        colnames = df.columns.to_list()

        seasonal_dfs, components = compute_seasonality_prophet(
            df_arg = df, date_column=date_column, value_columns=colnames, freq=freq
        )
        
        
        

        response = prepare_seasonality_response(
            df_arg=df, tsType=tsType, colnames=colnames, components = components,  seasonal_dfs= seasonal_dfs
        )

        return Response(json.dumps(response), mimetype="application/json")

@api.route('/hello', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World!"), 200



#==============================================================================================================
# import pandas as pd
# import numpy as np
# from flask import Flask, request, jsonify, Response, Blueprint
# from flask_cors import CORS
# import io
# import json




# api = Blueprint('api', __name__)




# @api.route('/hello', methods=['GET'])
# def hello_world():
#     return jsonify(message="Hello, World!"), 200