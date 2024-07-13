import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from utility.lags import *

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
        indexType = request.form.get('indexType')
        freq = request.form.get('freq')
        steps = request.form.get('steps')

        df = pd.read_csv(file, index_col= 0, parse_dates=True)
        # enforce the frequency
        df = df.asfreq(freq)

        # extract the
        lag = sig_lag(df, 50, ts_type='univariate')
        
        result = forecast_uni(df, lag, steps, freq, forecast_method="without_refit")
        
        


@app.route("/handle_submit_multi_ts", methods=["POST"])
def handle_submit_multi_ts(): ...


if __name__ == "__main__":
    app.run(debug=True)
