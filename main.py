# Import necessary libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('RidgeModel.pkl', 'rb'))

# A sample list of locations; in practice, you should load these from your dataset
locations = ['1st Phase JP Nagar', '2nd Phase Judicial Layout', 'Sarjapur  Road', 'Electronic City', 'Whitefield', 'Marathahalli', 'other']

# Route for home page
@app.route('/')
def index():
   return render_template('index.html', locations=locations)

# Route to handle form submissions and return predictions
@app.route('/predict', methods=['POST'])
def predict():
   # Get data from the form
   location = request.form['location']
   bhk = int(request.form['bhk'])
   bath = int(request.form['bath'])
   total_sqft = float(request.form['total_sqft'])
   
   # Create the feature array
   input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], 
                             columns=['location', 'total_sqft', 'bath', 'bhk'])
   
   # Make prediction using the model pipeline
   prediction = model.predict(input_data)[0]

   # Return the result to the user
   return render_template('index.html', locations=locations, prediction_text=f'Predicted House Price: ₹{prediction:.2f} Lakhs')

if __name__ == "__main__":
   app.run(debug=True)


