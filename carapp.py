import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and preprocess data
data = pd.read_csv(r"C:\Users\adity\OneDrive\Desktop\Car price prediction\car data (1).csv")
current_year = 2025
data['Car_Age'] = current_year - data['Year']
data = data.drop(['Car_Name', 'Year'], axis=1)
data = pd.get_dummies(data, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Car Price Prediction App")
st.write("Enter the details of the car to predict its price.")

# Input fields for user data
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
driven_kms = st.number_input("Driven Kilometers", min_value=0, step=100)
owner = st.selectbox("Number of Owners", [0, 1, 2, 3])
car_age = st.number_input("Car Age (in years)", min_value=0, step=1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

# Encode inputs
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0
seller_type_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# Predict price
if st.button("Predict Price"):
    features = np.array([[present_price, driven_kms, owner, car_age, fuel_type_diesel,
                          fuel_type_petrol, seller_type_individual, transmission_manual]])
    prediction = model.predict(features)
    st.success(f"Estimated Selling Price: ₹{prediction[0]:,.2f} lakhs")