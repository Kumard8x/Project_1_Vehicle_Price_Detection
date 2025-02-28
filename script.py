#loading required library
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lists import *



# Load trained model
rfr_model = joblib.load('Random_forest_regressor.pkl')

# Load onehot and Target encoders
onehot_encoder = joblib.load("Onehot_encoder.pkl")
target_encoder = joblib.load("Target_encoder.pkl")

# Title and description
st.title("🚘 Car Price Prediction App")
st.write("Enter car details to predict the price.")

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    # User input values fields
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015, step=1)
    cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4, step=1)
    mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=50000, step=1000)
    doors = st.number_input("Doors", min_value=2, max_value=6, value=4, step=1)
    make = st.selectbox("Make", Make)
    fuel = st.selectbox("Fuel", Fuel)
    transmission = st.selectbox('Transmission', Transmission)
with col2: 
    body = st.selectbox('Body', Body)
    drivetrain = st.selectbox('Drivetrain', Drivetrain)
    model = st.selectbox('Model', Model )
    engine = st.selectbox('Engine', Engine)
    trim = st.selectbox('Trim', Trim)
    exterior_color = st.selectbox('Exterior Color', Exterior_color)
    interior_color = st.selectbox('Interior Color', Interior_color)

    
    #selecting columns for one - hot encoding (low cardinality Nominal)
    onehot_cols = ['make', 'fuel', "transmission", 'body', 'drivetrain']

    #select colunms for target encoding (high cardinality Nominal)
    target_cols = ['model', 'engine', 'trim', 'exterior_color', 'interior_color']
    
# Create DataFrame for prediction
user_input_data = pd.DataFrame([[year, cylinders, mileage, doors, make, fuel, transmission, body, drivetrain, model, engine, 
                                 trim, exterior_color, interior_color]],
                               columns=['year', 'cylinders', 'mileage', 'doors', 'make', 'fuel', "transmission", 'body', 'drivetrain',
                                   'model', 'engine', 'trim', 'exterior_color', 'interior_color'])
                           


onehot_encoded_col = onehot_encoder.transform(user_input_data[onehot_cols])
onehot_encoded_df = pd.DataFrame(onehot_encoded_col, columns=onehot_encoder.get_feature_names_out(onehot_cols))
#
target_encoded_col = target_encoder.transform(user_input_data[target_cols])
user_input_data.drop(columns=['make', 'fuel', "transmission", 'body', 'drivetrain','model', 'engine', 
                                            'trim', 'exterior_color', 'interior_color'], inplace=True)

# Create DataFrame for prediction
input_data = pd.concat([user_input_data, onehot_encoded_df, target_encoded_col], axis=1)

# Make prediction
if st.button("Predict Price"):
    predicted_price = rfr_model.predict(input_data)[0]
    st.success(f"Estimated Car Price: ${predicted_price}")

    
st.markdown(""" 
    --- 
    ⚙️ Build By **Deepak Kumar** \n
    📩 **Contact Me:** 
    🔗 [LinkedIn](https://www.linkedin.com/in/deepak-kumar8/) 
    🔗 [GitHub](https://github.com/Kumard8x)
    📧 Email: deepak.kumar030151@gmail.com
    
    ---
    """)


