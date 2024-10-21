import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pre-trained model
model_path = 'crop_yield_model (1).pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the training columns (the columns generated during model training)
# These columns should match the ones used during the model's training process
with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to make predictions from form data
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        crop = request.form['Crop']
        area = float(request.form['Area'])
        production = float(request.form['Production'])
        annual_rainfall = float(request.form['Annual_Rainfall'])
        fertilizer = float(request.form['Fertilizer'])
        pesticide = float(request.form['Pesticide'])

        # Create input DataFrame
        input_data = pd.DataFrame([[crop, area, production, annual_rainfall, fertilizer, pesticide]],
                                  columns=['Crop', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

        # Apply one-hot encoding to the 'Crop' feature
        input_data = pd.get_dummies(input_data)

        # Reindex the columns so they match the training data (this will add missing columns with 0s)
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)

        # Render the form with the prediction result
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)



