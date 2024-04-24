# -*- coding: utf-8 -*-
"""  Flask REST API for diamond price predictions.

This script builds a Flask REST API (an "app") for making predictions of diamond
prices from a machine learning model.

The model object can be loaded from a local file (e.g. models/pipeline.joblib)
or from a URL. The app expects a POST request containing a JSON payload with the
following fields: 'carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color',
'clarity'. The payload can contain more than one item. The response is in JSON
format with a field 'price' that contains a list with the prediction/s.

The following third-party packages are required: pandas, dill, flask, and
requests.

The app can be started from the project's root directory with the following
command:

    $ python src/app.py

By default it listens to port 5000 in all IP addresses (0.0.0.0) and runs in
debug mode. It can be tested locally using the script src/request.py and the
payload data in the file res/payload.json.

Contains the following functions:

    * predict - process the POST request and return predictions.


@author: Patricio Carnelli
@contact: pcarnelli@gmail.com
@credit: Muhammad Bilal Shinwari
@links: https://github.com/Bilal-Shinwari/FLASK-REST-API-ML
@license: MIT
@date: 24-Apr-2024
@version: 0.1
"""


import tempfile
from typing import Any

import pandas as pd
import dill
import flask
import requests


# Create Flask app
app = flask.Flask(__name__)

# Load model from local file
#model = dill.load(open('models/pipeline.joblib', 'rb'))

# Load model from remote file
# This URL would give the last trained model with the github actions workflow
url = 'https://github.com/pcarnelli/xtream-ai-assignment-engineer/raw/test/models/pipeline.joblib'
response = requests.get(url)
with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
    temp_file.write(response.content)
model = dill.load(open(temp_file.name, 'rb'))
temp_file.close()


@app.route('/predict', methods=['POST'])
def predict() -> Any:

    """Returns model predictions for diamond prices from an API POST request
    that sends a data payload in JSON format.

    Returns:
        Any: Response with predictions in JSON format.
    """
    
    # Get JSON data from the request
    request_json = flask.request.get_json()

    # Convert JSON data into a dataframe
    df = pd.DataFrame.from_records(request_json)

    # Use loaded model to make predictions
    prediction = model.predict(df).astype(int).tolist()

    # Return prediction as JSON response
    return flask.jsonify({'price': prediction})


if __name__ == '__main__':

    """Run Flask app when this script is executed.
    """

    # Local debugging
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    # Production
    #app.run(host='0.0.0.0', port=5000)
