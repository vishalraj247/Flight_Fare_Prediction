import streamlit as st
from src.data.data_preprocessor import DataPreprocessor
import joblib
import tensorflow as tf

# Set Streamlit app title
st.title("Flight Fare Prediction")

# Collect user inputs via Streamlit
user_input = {
    'startingAirport': st.text_input('Enter origin airport:'),
    'destinationAirport': st.text_input('Enter destination airport:'),
    'flightDate': st.date_input('Select flight date:'),
    'segmentsDepartureTimeRaw': st.time_input('Select departure time:'),
    'segmentsCabinCode': st.selectbox('Choose cabin type:', options=['coach' 'premium coach' 'first' 'business'])
}

# Create a "Predict" button
if st.button("Predict"):
    # Preprocess user input
    data_preprocessor = DataPreprocessor()
    preprocessed_input = data_preprocessor.preprocess_user_input(
        user_input,
        'models/preprocessor.joblib',
        'models/category_mappings.joblib',
        'data/processed/avg_features.csv'
    )

    # Paths to all the students' models
    model_paths = [
        "models/my_final_model.keras",
    #    "models/student2_model.keras",
    #    "models/student3_model.keras",
    #    "models/student4_model.keras"
    ]

    # Loop through each model, predict and display results
    for idx, model_path in enumerate(model_paths, 1):
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        
        # Use preprocessed_input for model prediction
        predicted_fare = model.predict(preprocessed_input)

        # Display the predicted fare
        st.write(f"Prediction from Model {idx}: ${predicted_fare[0][0]:.2f}")