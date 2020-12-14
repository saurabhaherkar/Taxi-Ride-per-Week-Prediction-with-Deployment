import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('Taxi.pkl')


@app.route('/')
def home():
    return render_template('index_taxi.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    feature_values = [np.array(input_features)]
    prediction = model.predict(feature_values)
    out = round(prediction[0], 2)
    return render_template('index_taxi.html', prediction_text='The number of weekly rides should be: {}'.format(out))


if __name__ == '__main__':
    app.run(debug=True)
