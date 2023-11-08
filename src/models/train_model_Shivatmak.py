import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Flatten, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

class LSTMModel:
    def __init__(self, data, categorical_features, numerical_features, target_column):
        self.data = data
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_column = target_column
        self.model = None
        self.compute_embedding_sizes()

    def compute_embedding_sizes(self):
        self.embedding_sizes = {
            feature: (self.data[feature].max() + 1, min(50, (self.data[feature].nunique() + 1) // 2))
            for feature in self.categorical_features
        }

    def preprocess_data(self):
        # Separate features and target variable
        self.categorical_data = self.data[self.categorical_features]
        self.numerical_data = self.data[self.numerical_features]
        self.target = self.data[self.target_column]

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=42)

    def build_model(self):
        # List to hold all inputs
        inputs = []
        embeddings = []

        # Process categorical features
        for feature in self.categorical_features:
            input_layer = Input(shape=(1,))
            inputs.append(input_layer)
            embedding_layer = Embedding(input_dim=self.embedding_sizes[feature][0],
                                        output_dim=self.embedding_sizes[feature][1],
                                        input_length=1)(input_layer)
            embedding_layer = Flatten()(embedding_layer)
            embeddings.append(embedding_layer)

        # Process numerical features
        numerical_input = Input(shape=(len(self.numerical_features),))
        inputs.append(numerical_input)

        # Concatenate all embeddings along with numerical input
        concatenated_layers = Concatenate()(embeddings + [numerical_input])

        # Add a Reshape layer to add the time step dimension
        reshaped_layer = Reshape((1, -1))(concatenated_layers)

        # Define the LSTM structure
        lstm_out = LSTM(50, activation='relu')(reshaped_layer)
        lstm_out = Dropout(0.2)(lstm_out)
        output = Dense(1)(lstm_out)

        # Create the model
        self.model = Model(inputs=inputs, outputs=output)

        # Compile the model with the additional metrics
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=[MeanSquaredError(), MeanAbsoluteError()]
        )

    def train_model(self, epochs=100, batch_size=1024, callbacks=None):  # Updated batch_size
        # Prepare the inputs for the model
        X_train_inputs = [self.X_train[feature].values for feature in self.categorical_features]
        X_train_inputs.append(self.X_train[self.numerical_features].values)

        # Train the model
        self.model.fit(
            X_train_inputs, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.1, 
            callbacks=callbacks
        )

    def evaluate_model(self, batch_size=1024):
        # Prepare the inputs for the model
        X_test_inputs = [self.X_test[feature].values for feature in self.categorical_features]
        X_test_inputs.append(self.X_test[self.numerical_features].values)

        # Evaluate the model and return a dictionary of the metrics
        evaluation_results = self.model.evaluate(
            X_test_inputs, 
            self.y_test, 
            batch_size=batch_size,
            verbose=1,  # You can set this to 0 to reduce log verbosity
            return_dict=True  # This will return the metrics as a dictionary
        )
        return evaluation_results