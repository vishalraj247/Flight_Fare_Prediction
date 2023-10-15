import streamlit as st
import joblib
import pandas as pd

# Load saved preprocessing objects
# Preprocessor.joblib is a saved ColumnTransformer object
preprocessor = joblib.load('path_to_saved_preprocessor.joblib')

# Categorical_mappings.joblib contains mappings from categories to integers
category_mappings = joblib.load('path_to_categorical_mappings.joblib')

# Define a function to preprocess user input
def preprocess_user_input(user_input, preprocessor, category_mappings):
    user_input_df = pd.DataFrame([user_input])
    
    # Convert categorical variables to integers using saved mappings
    for col, mapping in category_mappings.items():
        user_input_df[col] = user_input_df[col].map(mapping)
    
    # Apply the preprocessor
    preprocessed_input = preprocessor.transform(user_input_df)
    
    return preprocessed_input

# Assume user_input is a dictionary of inputs from the user via Streamlit
user_input = {
    'startingAirport': st.text_input('Enter origin airport:'),
    'destinationAirport': st.text_input('Enter destination airport:'),
    'flightDate': st.date_input('Select flight date:'),
    'segmentsDepartureTimeRaw': st.time_input('Select departure time:'),
    'segmentsCabinCode': st.selectbox('Choose cabin type:', options=['coach', 'premium'])
}

# Preprocess the user input
preprocessed_input = preprocess_user_input(user_input, preprocessor, category_mappings)

# The preprocessed_input can now be used as input for our model
