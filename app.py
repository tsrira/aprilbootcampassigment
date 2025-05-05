import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('xgboost_classification.pkl')  # Replace with the actual path to your saved model

# Load the new CSV file
car_data = pd.read_csv('cars_brand_model.csv')  # Path to your uploaded CSV file

# Clean the data (remove leading/trailing whitespaces, handle missing values)
car_data['Brand'] = car_data['Brand'].str.strip()
car_data['Model'] = car_data['Model'].str.strip()
car_data = car_data.dropna(subset=['Brand', 'Model'])  # Drop rows with missing Brand or Model

# Initialize LabelEncoders for Brand and Model
brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()
label_encoder = LabelEncoder()

# Fit encoders (assuming the model was trained with these encoders)
car_data['Brand'] = brand_encoder.fit_transform(car_data['Brand'])
car_data['Model'] = model_encoder.fit_transform(car_data['Model'])
car_data['Fuel_Type'] = label_encoder.fit_transform(input_data['Fuel_Type'])
car_data['Transmission'] = label_encoder.fit_transform(input_data['Transmission'])

# Get unique values for brand and model from the dataset
brands = car_data['Brand'].unique()
models = car_data['Model'].unique()

# Function to make predictions
def predict_price(brand, model_input, year, engine_size, fuel_type, transmission, mileage, doors, owner_count):
    # Encode brand and model before passing to the model
    brand_encoded = brand_encoder.transform([brand])[0]
    model_encoded = model_encoder.transform([model_input])[0]
    fuel_type_encoded = label_encoder.transform([fuel_type])[0]
    transmission_encoded = label_encoder.transform([transmission])[0]
    
    
    # Prepare the input data
    input_data = pd.DataFrame({
        'Brand': [brand_encoded],
        'Model': [model_encoded],
        'Year': [year],
        'Engine_Size': [engine_size],
        'Fuel_Type': [fuel_type_encoded],  # Pass original Fuel_Type for input
        'Transmission': [transmission_encoded],  # Pass original Transmission for input
        'Mileage': [mileage],
        'Doors': [doors],
        'Owner_Count': [owner_count]
    })
    
   # One-hot encode fuel_type and transmission columns (make sure it matches the training data)
   # input_data = pd.get_dummies(input_data, columns=['Fuel_Type', 'Transmission'], drop_first=True)
    
    # Ensure the feature columns match the training data
    missing_columns = set(car_data.columns) - set(input_data.columns)
    for col in missing_columns:
        input_data[col] = 0  # Fill missing columns with zeros (or any appropriate default value)
    
    # Make the prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title("Car Price Prediction")

# Input fields for user using selectbox
brand = st.selectbox("Select Car Brand", brand_encoder.classes_)
model_input = st.selectbox("Select Car Model", model_encoder.classes_)

year = st.number_input("Car Year", min_value=1900, max_value=2023, value=2020)
engine_size = st.number_input("Engine Size", min_value=0.5, max_value=8.0, value=2.0)
fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'Hybrid', 'Electric'])  # Use original categories
transmission = st.selectbox("Transmission", ['Manual', 'Automatic', 'Semi-Automatic'])  # Use original categories
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
