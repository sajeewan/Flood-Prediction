from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/logistic_regression_model.joblib')
scaler = joblib.load('model/scaler.joblib')  # Load the scaler used for training

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        river_water_level = float(request.form['river_water_level'])

        # Scale the input features
        user_input = scaler.transform([[rainfall, temperature, river_water_level]])

        # Make a prediction
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input)[0, 1]

        return render_template('index.html', prediction=prediction[0], probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
