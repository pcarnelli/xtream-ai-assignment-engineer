from typing import Any
import pandas as pd
import dill
import flask


# TODO: Docstrings


# Creating a Flask app
app = flask.Flask(__name__)


# Define function for making predictions
@app.route('/predict', methods=['POST'])
def predict() -> Any:

    # Load model from joblib file
    model = dill.load(open('models/pipeline.joblib', 'rb'))

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
    app.run(debug=True)
