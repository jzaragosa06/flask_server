import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

from flask import Response
import json


from utility.lags import *
from forecast.forecast_uni import *
from forecast.forecast_uni_with_gap import *
from forecast.forecast_multi import *
from forecast.forecast_multi_with_gap import *
from utility.gap_functions import *
from Trend_analysis.sma import *
from seasonality_analysis.seasonal import *
from utility.prepare_response import *


app = Flask(__name__)
# this will disable the CORS protection for the proceeding routes.
CORS(app)


@app.route("/handle_submit", methods=["POST"])
def handle_submit():
    if "inputFile" not in request.files:
        print("error1")
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files["inputFile"]

    if file.filename == "":
        print("error2")
        return jsonify({"message": "No selected file"}), 400

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
        window_size = request.form.get("window_size")
        hasSeasonal = request.form.get("seasonal")
        steps = request.form.get("steps")
        forecastMethod = request.form.get("method")

        # df = pd.read_csv(file, index_col=0, parse_dates=True, encoding="utf-8")
        # # we don't want to enforce the frequency since it will fill the dates in between gaps.
        # # enforce the frequency
        # # df = df.asfreq(freq)
        # print(df)

        hasGap = checkGap(df=df, freq=freq)
        print("has gap? " + str(hasGap))

        # we need a sufficient reason to the number we will pass on the second parameter.
        if tsType == "univariate":
            lag = sig_lag(df, 50, ts_type="univariate")

            # extract trend and seasonality behaviour of the ts data.
            trend_result = compute_sma(
                df_arg=df, window_size=int(window_size), ts_type=tsType
            )
            seasonal_result = compute_seasonality(df_arg=df, ts_type=tsType)

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
                    print(f"predicted.train on all: {pred_out}")
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
                    print(f"predicted.train on all: {pred_out}")
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

            response_json = prepare_json_response(
                df_arg=df,
                tsType=tsType,
                freq=freq,
                description=description,
                window_size=window_size,
                hasSeasonal=hasSeasonal,
                steps=steps,
                forecastMethod=forecastMethod,
                trend_result=trend_result,
                seasonal_result=seasonal_result,
                metric=metric,
                pred_test=pred_test,
                pred_out=pred_out,
            )
            print(response_json)
            return Response(json.dumps(response_json), mimetype="application/json")

        else:
            dict_lags, lag_list = sig_lag(df, 50, ts_type="multivariate")

            trend_result = compute_sma(
                df_arg=df, window_size=int(window_size), ts_type=tsType
            )
            seasonal_result = compute_seasonality(df_arg=df, ts_type=tsType)

            if hasGap == False:
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

            print(pred_out)
            print(trend_result)
            print(seasonal_result)

    try:
        response_json = prepare_json_response(
            df_arg=df,
            tsType=tsType,
            freq=freq,
            description=description,
            window_size=window_size,
            hasSeasonal=hasSeasonal,
            steps=steps,
            forecastMethod=forecastMethod,
            trend_result=trend_result,
            seasonal_result=seasonal_result,
            metric=metric,
            pred_test=pred_test,
            pred_out=pred_out,
        )

        print(response_json)
        # return jsonify(response_json)

        
        return Response(json.dumps(response_json), mimetype="application/json")
    except Exception as e:
        print(f"Error in preparing response: {e}")
        return jsonify({"message": f"Error in preparing response: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
