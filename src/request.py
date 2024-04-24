import requests
import json


# TODO: Docstrings


# Define the URL of your Flask API
url = 'http://0.0.0.0:5000/predict'

# Open input data as JSON file
with open('res/payload.json') as f:
    data = json.load(f)

# Send a POST request to the API with the input data
response = requests.post(url, json=data)

# Check the HTTP response status code
if response.status_code == 200:
    # Parse and print the JSON response (assuming it contains the prediction)
    prediction = response.json()
    json.dump(prediction, open(f'res/prediction.json', 'w'))
    print(prediction)
else:
    # Handle the case where the API request failed
    print(f'API Request Failed with Status Code: {response.status_code}')
    print(f'Response Content: {response.text}')
