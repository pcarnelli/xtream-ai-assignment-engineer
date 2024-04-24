from typing import Any
import pandas as pd
import dill
import flask
import tempfile
import requests


# TODO: Docstrings


# Create Flask app
app = flask.Flask(__name__)

# Load model from local file
#model = dill.load(open('models/pipeline.joblib', 'rb'))

# Load model from remote file
url = 'https://github.com/pcarnelli/xtream-ai-assignment-engineer/raw/test/models/pipeline.joblib'
response = requests.get(url)
with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
    temp_file.write(response.content)
model = dill.load(open(temp_file.name, 'rb'))
temp_file.close()


# Define function for making predictions
@app.route('/predict', methods=['POST'])
def predict() -> Any:

    # Get JSON data from the request
    request_json = flask.request.get_json()

    # Convert JSON data into a DataFrame
    df = pd.DataFrame.from_records(request_json)

    # Use loaded model to make predictions
    prediction = model.predict(df).astype(int).tolist()

    # Return prediction as JSON response
    return flask.jsonify({'price': prediction})


# Run Flask app when this script is executed
if __name__ == '__main__':
    
    # Local debugging
	app.run(debug=True, host='0.0.0.0',port=5000)
    
    # Production
    #app.run(host='0.0.0.0', port=5000)
