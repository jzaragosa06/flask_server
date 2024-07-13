import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from utility.lags import *
from forecast.forecast_univariate import *

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
        # extract the other variables.
        indexType = request.form.get("indexType")
        freq = request.form.get("freq")
        steps = request.form.get("steps")
        forecastMethod = request.form.get("forecastMethod")

        df = pd.read_csv(file, index_col=0, parse_dates=True)
        # enforce the frequency
        df = df.asfreq(freq)

        print(df)

        # extract the
        lag = sig_lag(df, 50, ts_type="univariate")
        print(lag)

        if forecastMethod == "without_refit":
            result = forecast_uni(
                df, int(lag), int(steps), freq, forecast_method="without_refit"
            )
            print(f"predicted.train on all: {result}")

            metric, pred = evaluate_model(
                df, int(lag), int(steps), freq, forecast_method="without_refit"
            )

            print(f"mse on training data: {metric}. predicted value on test: {pred}")
        else:
            result = forecast_uni(
                df, int(lag), int(steps), freq, forecast_method="with_refit"
            )
            print(f"predicted.train on all: {result}")

            metric, pred = evaluate_model(
                df, int(lag), int(steps), freq, forecast_method="with_refit"
            )

            print(f"mse on training data: {metric}. predicted value on test: {pred}")


@app.route("/handle_submit_multi_ts", methods=["POST"])
def handle_submit_multi_ts(): ...


if __name__ == "__main__":
    app.run(debug=True)
