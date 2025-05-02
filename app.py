import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load the trained model (Make sure to replace 'path_to_your_model.pkl' with the actual model file path)
model = joblib.load('xgboost_classification.pkl')  # Replace with the actual path to your saved model

# Load the car dataset
car_data = pd.read_csv('/mnt/data/cars_price.csv')  # Make sure the path to your uploaded CSV is correct

# Get unique values for brand and model from the dataset
brands = car_data['Brand'].unique()
models = car_data['Model'].unique()

# Function to make predictions
def predict_price(brand, model_input, year, engine_size, fuel_type, transmission, mileage, doors, owner_count):
    # Prepare the input data
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Model': [model_input],
        'Year': [year],
        'Engine_Size': [engine_size],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Mileage': [mileage],
        'Doors': [doors],
        'Owner_Count': [owner_count]
    })
    
    # Make the prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title("Car Price Prediction")

# Input fields for user
brand = st.selectbox("Select Car Brand", brands)
model_input = st.selectbox("Select Car Model", models)

year = st.number_input("Car Year", min_value=1900, max_value=2023, value=2020)
engine_size = st.number_input("Engine Size", min_value=0.5, max_value=8.0, value=2.0)
fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'Hybrid', 'Electric'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic', 'Semi-Automatic'])
mileage = st.number_input("Mileage (in km)", min_value=0, value=50000)
doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=4)
owner_count = st.number_input("Number of Owners", min_value=1, value=1)

# Prediction Button
if st.button("Predict"):
    if brand and model_input:
        price = predict_price(brand, model_input, year, engine_size, fuel_type, transmission, mileage, doors, owner_count)
        st.success(f"The predicted price of the car is: ₹{price:.2f}")
    else:
        st.error("Please fill in all the required fields")

