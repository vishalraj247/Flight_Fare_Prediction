import streamlit as st
from src.data.data_preprocessor import DataPreprocessor
import joblib

# Collect user inputs via Streamlit
user_input = {
    'startingAirport': st.text_input('Enter origin airport:'),
    'destinationAirport': st.text_input('Enter destination airport:'),
    'flightDate': st.date_input('Select flight date:'),
    'segmentsDepartureTimeRaw': st.time_input('Select departure time:'),
    'segmentsCabinCode': st.selectbox('Choose cabin type:', options=['coach', 'premium'])
}

# Preprocess user input
data_preprocessor = DataPreprocessor()
preprocessed_input = data_preprocessor.preprocess_user_input(
    user_input,
    'path_to_saved_preprocessor.joblib',
    'path_to_saved_category_mappings.joblib',
    'path_to_save_avg_features.csv'
)

# Load your trained model
model = joblib.load('path_to_saved_model.joblib')

# Use preprocessed_input for model prediction
predicted_fare = model.predict(preprocessed_input)

# Display the predicted fare
st.write(f"The estimated fare is: ${predicted_fare[0]:.2f}")