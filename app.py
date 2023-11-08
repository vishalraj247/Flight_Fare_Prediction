import streamlit as st
from src.data.data_preprocessor import DataPreprocessor
import joblib
import tensorflow as tf
import pandas as pd

# Set Streamlit app title
st.title("Flight Fare Prediction")

# List of airports
airports_list = ['ATL', 'MIA', 'PHL', 'SFO', 'LGA', 'LAX', 'ORD', 'IAD', 'EWR', 'DEN', 'DFW', 'BOS', 'OAK', 'DTW', 'CLT', 'JFK']

# Collect user inputs via Streamlit
user_input = {
    'startingAirport': st.selectbox('Select origin airport:', options=airports_list),
    'destinationAirport': st.selectbox('Select destination airport:', options=airports_list),
    'flightDate': st.date_input('Select flight date:'),
    'segmentsDepartureTimeRaw': st.time_input('Select departure time:'),
    'segmentsCabinCode': st.selectbox('Choose cabin type:', options=['coach', 'premium coach', 'first', 'business'])
}

# Create a "Predict" button
if st.button("Predict"):
    # Preprocess user input
    data_preprocessor = DataPreprocessor()
    preprocessed_input = data_preprocessor.preprocess_user_input(
        user_input,
        'models/preprocessor_dl.joblib',
        'models/category_mappings_dl.joblib',
        'data/processed/avg_features_dl.csv'
    )

    # Paths to all the students' models
    model_student_mapping = {
        "models/best_model-vishal_raj": "Vishal Raj's Model",
        "models/best_model_Shivatmak": "Shivatmak's Model",
    #    "models/ronik_model": "Ronik's Model",
    #    "models/student4_model": "Student 4's Model"
    }

    # Loop through each model, predict and display results
    for model_path, student_name in model_student_mapping.items():
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        # Identify and print the input shapes
 #       for input_tensor in model.inputs:
 #           st.write(f"Input tensor shape: {input_tensor.shape}")
 #       st.dataframe(preprocessed_input)

        # 1. Other wide features
        wide_features = preprocessed_input[['flightDate_year', 'flightDate_month', 'flightDate_day', 'flightDate_weekday', 'flightDate_is_weekend', 'segmentsDepartureTimeRaw_hour', 'segmentsDepartureTimeRaw_minute']].values

        # 2. startingAirport input
        startingAirport = preprocessed_input[['startingAirport']].values

        # 3. destinationAirport input
        destinationAirport = preprocessed_input[['destinationAirport']].values

        # 4. segmentsCabinCode input
        segmentsCabinCode = preprocessed_input[['segmentsCabinCode']].values

        # 5. Deep features
        deep_features = preprocessed_input[['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']].values

        # 6. Numerical features
        numerical_features = preprocessed_input[['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']].values

        # Predict and display results depending on model path
        if "vishal_raj" in model_path:
            predicted_fare = model.predict([wide_features, startingAirport, destinationAirport, segmentsCabinCode, deep_features])
            st.write(f"Prediction from {student_name}: ${predicted_fare[0][0]:.2f}")
        elif "Shivatmak" in model_path:
            predicted_fare1 = model.predict([startingAirport, destinationAirport, segmentsCabinCode, numerical_features])
            st.write(f"Prediction from {student_name}: ${predicted_fare1[0][0]:.2f}")
#        elif "Ronik" in model_path:
#            predicted_fare2 = 
#            st.write(f"Prediction from {student_name}: ${predicted_fare1[0][0]:.2f}")