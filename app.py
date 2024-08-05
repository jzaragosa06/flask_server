# import pandas as pd
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# from utility.lags import *
# from forecast.forecast_univariate import *

# app = Flask(__name__)
# # this will disable the CORS protection for the proceeding routes.
# CORS(app)


# @app.route("/handle_submit_uni_ts", methods=["POST"])
# def handle_submit_uni_ts():
#     if "inputFile" not in request.files:
#         print("error1")
#         return jsonify({"message": "No file part in the request"}), 400

#     file = request.files["inputFile"]

#     if file.filename == "":
#         print("error2")
#         return jsonify({"message": "No selected file"}), 400

#     if file:
#         # extract the other variables.
#         indexType = request.form.get("indexType")
#         freq = request.form.get("freq")
#         steps = request.form.get("steps")
#         forecastMethod = request.form.get("forecastMethod")

#         df = pd.read_csv(file, index_col=0, parse_dates=True)
#         # enforce the frequency
#         df = df.asfreq(freq)

#         print(df)

#         # extract the
#         lag = sig_lag(df, 50, ts_type="univariate")
#         print(lag)

#         if forecastMethod == "without_refit":
#             result = forecast_uni(
#                 df, int(lag), int(steps), freq, forecast_method="without_refit"
#             )
#             print(f"predicted.train on all: {result}")

#             metric, pred = evaluate_model(
#                 df, int(lag), int(steps), freq, forecast_method="without_refit"
#             )

#             print(f"mse on training data: {metric}. predicted value on test: {pred}")
#         else:
#             result = forecast_uni(
#                 df, int(lag), int(steps), freq, forecast_method="with_refit"
#             )
#             print(f"predicted.train on all: {result}")

#             metric, pred = evaluate_model(
#                 df, int(lag), int(steps), freq, forecast_method="with_refit"
#             )

#             print(f"mse on training data: {metric}. predicted value on test: {pred}")


# @app.route("/handle_submit_multi_ts", methods=["POST"])
# def handle_submit_multi_ts(): ...


# if __name__ == "__main__":
#     app.run(debug=True)


import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from utility.lags import *
from forecast.forecast_uni import *
from forecast.forecast_uni_with_gap import *
from forecast.forecast_multi import *
from forecast.forecast_multi_with_gap import *
from utility.gap_functions import *

app = Flask(__name__)
# this will disable the CORS protection for the proceeding routes.
CORS(app)


@app.route("/handle_submit_uni_ts", methods=["POST"])
def handle_submit_uni_ts():
    if "inputFile" not in request.files:
        print("error1")
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files["inputFile"]

    if file.filename == "":
        print("error2")
        return jsonify({"message": "No selected file"}), 400

    if file:
        indexType = request.form.get("indexType")
        freq = request.form.get("freq")
        steps = request.form.get("steps")
        forecastMethod = request.form.get("forecastMethod")
        tsType = request.form.get("type")
        # we need a function that will identify whether a ts data has a gap or not.
        # this function returns a bool.
        hasGap = checkGap(df_arg=df)

        df = pd.read_csv(file, index_col=0, parse_dates=True)
        # we don't want to enforce the frequency since it will fill the dates in between gaps.
        # enforce the frequency
        # df = df.asfreq(freq)
        print(df)

        # we need a sufficient reason to the number we will pass on the second parameter.
        if tsType == "univariate":
            lag = sig_lag(df, 50, ts_type="univariate")

            if forecastMethod == "without_refit":
                if hasGap == False:
                    metric, pred_test = evaluate_model_uni(df_arg=df, lag_value=int(lag), steps_value=int(steps), freq=freq, forecast_method="without_refit"
                                                           )

                    pred_out = forecast_uni(
                        df_arg=df, lag_value=int(lag), steps_value=int(steps), freq=freq, forecast_method="without_refit"
                    )
                    print(f"predicted.train on all: {pred_out}")
                else:
                    metric, pred_test = evaluate_model_uni(
                        df_arg=df, lag_value=int(lag), steps_value=int(steps), freq=freq, forecast_method="with_refit"
                    )

                    pred_out = forecast_uni(
                        df_arg=df, lag_value=int(lag), steps_value=int(steps), freq=freq, forecast_method="with_refit"
                    )
                    print(f"predicted.train on all: {pred_out}")
            else:
                if hasGap == False:
                    gap_length, interval_length = identify_gap(
                        df=df, freq=freq)

                    metric, pred_test = forecast_uni_with_gap(df_arg=df, lag_value=int(
                        lag), steps_value=steps, freq=freq, gap_length=gap_length, interval_length_before_gap=interval_length, forecast_method="with_refit")

                    pred_out = evaluate_model_uni_with_gap(df_arg=df, lag_value=int(
                        lag), steps_value=steps, freq=freq, gap_length=gap_length, interval_length_before_gap=interval_length, forecast_method="with_refit")

                else:
                    gap_length, interval_length = identify_gap(
                        df=df, freq=freq)

                    metric, pred_test = forecast_uni_with_gap(df_arg=df, lag_value=int(
                        lag), steps_value=steps, freq=freq, gap_length=gap_length, interval_length_before_gap=interval_length, forecast_method="without_refit")

                    pred_out = evaluate_model_uni_with_gap(df_arg=df, lag_value=int(
                        lag), steps_value=steps, freq=freq, gap_length=gap_length, interval_length_before_gap=interval_length, forecast_method="without_refit")

        else:
            dict_lags, lag_list = sig_lag(df, 50, ts_type="multivariate")
            if forecastMethod == "without_refit":
                if hasGap == False:
                    metric, pred_test = evaluate_model_multi(
                        df_arg=df, dict_lags=dict_lags, steps_value=steps, freq=freq, forecast_method="without_refit")

                    pred_out = forecast_multi(
                        df_arg=df, dict_lags=dict_lags, steps_value=steps, freq=freq, forecast_method="without_refit")
                    
                else:
                    ...
            else:
                ...
                if hasGap == False:
                    ...
                else:
                    ...


@app.route("/handle_submit_multi_ts", methods=["POST"])
def handle_submit_multi_ts(): ...


if __name__ == "__main__":
    app.run(debug=True)
