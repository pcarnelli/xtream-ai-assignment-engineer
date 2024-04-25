# -*- coding: utf-8 -*-
"""  Request for testing the Flask REST API for diamond price predictions.

This script sends a POST request to the Flask REST API src/app.py, then receives
and process the response.

The request contains a JSON data payload (read from file res/payload.json) with
the following fields: 'carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color',
'clarity'. The payload can contain more than one item. The response is expected
to be in JSON format with a field 'price' that contains a list with the
prediction/s. If the request is successful (code 200), the response is printed
on screen and saved to disk (file res/prediction.json).

The following third-party packages are required: requests.

While the API is up, the script can be executed from the project's root
directory with the following command:

    $ python src/request.py

By default it sends the request to port 5000 of the local host (127.0.0.1).


@author: Patricio Carnelli
@contact: pcarnelli@gmail.com
@credit: Muhammad Bilal Shinwari
@links: https://github.com/Bilal-Shinwari/FLASK-REST-API-ML
@license: MIT
@date: 24-Apr-2024
@version: 0.1
"""


import json

import requests


# Define the URL of the Flask REST API
url = 'http://127.0.0.1:5000/predict'

# Open data payload as JSON file
with open('res/payload.json') as f:
    data = json.load(f)

# Send a POST request to the API with the data payload in JSON format
response = requests.post(url, json=data)

# Check the HTTP response status code
if response.status_code == 200:
    # Parse and print the JSON response (assuming it contains the prediction)
    prediction = response.json()
    json.dump(prediction, open('res/prediction.json', 'w'))
    print(prediction)
else:
    # Handle the case where the API request failed
    print(f'API request failed with status code: {response.status_code}')
    print(f'Response content: {response.text}')
