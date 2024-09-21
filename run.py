import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import io
import json
from app import create_app

app = create_app()
CORS(app)

if __name__ == "__main__":
    app.run(debug=True)
