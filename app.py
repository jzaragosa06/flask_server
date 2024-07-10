import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/handle_submit_uni_ts", methods=["POST"])
def handle_submit_uni_ts(): ...


@app.route("/handle_submit_multi_ts", methods=["POST"])
def handle_submit_multi_ts(): ...


if __name__ == "__main__":
    app.run(debug=True)
