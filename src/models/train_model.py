import os
import pandas as pd
import tensorflow as tf

# 1. Load the data
def load_and_combine_data(base_path='data/processed'):
    # List all subdirectories (each subdirectory corresponds to an airport code)
    airport_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # List comprehension to load each CSV and concatenate them
    all_data = pd.concat([pd.read_csv(os.path.join(base_path, airport, f"{airport}_processed.csv")) for airport in airport_dirs], ignore_index=True)
    
    return all_data

# 2. Define the wide & deep neural network architecture
def build_model(embedding_sizes):
    # Wide component
    wide_input = tf.keras.layers.Input(shape=(7,))  # 7 wide features excluding the 3 categorical features
    
    # Embedding layers for categorical features
    startingAirport_input = tf.keras.layers.Input(shape=(1,))
    startingAirport_embedding = tf.keras.layers.Embedding(embedding_sizes['startingAirport'][0], embedding_sizes['startingAirport'][1])(startingAirport_input)
    startingAirport_embedding = tf.keras.layers.Flatten()(startingAirport_embedding)
    
    destinationAirport_input = tf.keras.layers.Input(shape=(1,))
    destinationAirport_embedding = tf.keras.layers.Embedding(embedding_sizes['destinationAirport'][0], embedding_sizes['destinationAirport'][1])(destinationAirport_input)
    destinationAirport_embedding = tf.keras.layers.Flatten()(destinationAirport_embedding)
    
    segmentsCabinCode_input = tf.keras.layers.Input(shape=(1,))
    segmentsCabinCode_embedding = tf.keras.layers.Embedding(embedding_sizes['segmentsCabinCode'][0], embedding_sizes['segmentsCabinCode'][1])(segmentsCabinCode_input)
    segmentsCabinCode_embedding = tf.keras.layers.Flatten()(segmentsCabinCode_embedding)
    
    # Combine wide features and embeddings
    wide_combined = tf.keras.layers.concatenate([wide_input, startingAirport_embedding, destinationAirport_embedding, segmentsCabinCode_embedding])
    
    # Deep component
    deep_input = tf.keras.layers.Input(shape=(3,))  # 3 deep features
    deep_layer1 = tf.keras.layers.Dense(256, activation='relu')(deep_input)
    deep_dropout1 = tf.keras.layers.Dropout(0.2)(deep_layer1)
    deep_layer2 = tf.keras.layers.Dense(128, activation='relu')(deep_dropout1)
    deep_dropout2 = tf.keras.layers.Dropout(0.2)(deep_layer2)
    
    # Combine wide and deep components
    combined = tf.keras.layers.concatenate([wide_combined, deep_dropout2])
    
    output_layer = tf.keras.layers.Dense(1)(combined)
    
    model = tf.keras.models.Model(inputs=[wide_input, startingAirport_input, destinationAirport_input, segmentsCabinCode_input, deep_input], outputs=output_layer)
    
    return model