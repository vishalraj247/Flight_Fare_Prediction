import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data.data_preprocessor import DataPreprocessor

#from keras_tuner import HyperModel, Hyperband
#from tensorflow.keras import regularizers

# class WideDeepHyperModel(HyperModel):
#     def __init__(self, embedding_sizes):
#         self.embedding_sizes = embedding_sizes

#     def build(self, hp):
#         # Wide component
#         wide_input = tf.keras.layers.Input(shape=(7,))  # 7 wide features excluding the 3 categorical features
        
#         # Embedding layers for categorical features
#         startingAirport_input = tf.keras.layers.Input(shape=(1,))
#         startingAirport_embedding = tf.keras.layers.Embedding(self.embedding_sizes['startingAirport'][0], 
#                                                             self.embedding_sizes['startingAirport'][1])(startingAirport_input)
#         startingAirport_embedding = tf.keras.layers.Flatten()(startingAirport_embedding)
        
#         destinationAirport_input = tf.keras.layers.Input(shape=(1,))
#         destinationAirport_embedding = tf.keras.layers.Embedding(self.embedding_sizes['destinationAirport'][0], 
#                                                         self.embedding_sizes['destinationAirport'][1])(destinationAirport_input)
#         destinationAirport_embedding = tf.keras.layers.Flatten()(destinationAirport_embedding)
        
#         segmentsCabinCode_input = tf.keras.layers.Input(shape=(1,))
#         segmentsCabinCode_embedding = tf.keras.layers.Embedding(self.embedding_sizes['segmentsCabinCode'][0], 
#                                                         self.embedding_sizes['segmentsCabinCode'][1])(segmentsCabinCode_input)
#         segmentsCabinCode_embedding = tf.keras.layers.Flatten()(segmentsCabinCode_embedding)
        
#         # Combine wide features and embeddings
#         wide_combined = tf.keras.layers.concatenate([wide_input, startingAirport_embedding, destinationAirport_embedding, segmentsCabinCode_embedding])

#         # Deep component with tunable parameters
#         deep_input = tf.keras.layers.Input(shape=(3,))  
#         for i in range(hp.Int('num_layers', 2, 4)):
#             if i == 0:
#                 x = tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 64, 256, 32),
#                                           activation='relu',
#                                           kernel_regularizer=regularizers.l2(hp.Choice('reg_rate', [0.01, 0.001, 0.0001])))(deep_input)
#             else:
#                 x = tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 64, 256, 32),
#                                           activation='relu',
#                                           kernel_regularizer=regularizers.l2(hp.Choice('reg_rate', [0.01, 0.001, 0.0001])))(x)
#             x = tf.keras.layers.Dropout(rate=hp.Float('dropout', 0.1, 0.5, step=0.1))(x)

#         # Combine wide and deep components
#         combined = tf.keras.layers.concatenate([wide_combined, x])
        
#         output_layer = tf.keras.layers.Dense(1)(combined)

#         model = tf.keras.models.Model(inputs=[wide_input, startingAirport_input, destinationAirport_input, segmentsCabinCode_input, deep_input], outputs=output_layer)
#         model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')), loss='mean_squared_error', metrics=['mae', 'mse'])

#         return model

class WideDeepModel:
    
    def __init__(self, processed_data, preprocessor, avg_features):
        self.embedding_sizes = None
        self.model = None
        self.preprocessor = preprocessor
        self.avg_features = avg_features
        self.data = processed_data
        self.compute_embedding_sizes()

#    @staticmethod
#    def load_merged_data(filename='merged_data_processed_dl.csv', base_path='data/processed'):
#        data_path = os.path.join(base_path, filename)
#        return pd.read_csv(data_path)

    def compute_embedding_sizes(self):
        self.embedding_sizes = {
            'startingAirport': (self.data['startingAirport'].nunique() + 1, min(50, (self.data['startingAirport'].nunique() + 1) // 2)),
            'destinationAirport': (self.data['destinationAirport'].nunique() + 1, min(50, (self.data['destinationAirport'].nunique() + 1) // 2)),
            'segmentsCabinCode': (self.data['segmentsCabinCode'].nunique() + 1, min(50, (self.data['segmentsCabinCode'].nunique() + 1) // 2))
        }

    def build_model(self):
        # Wide component
        wide_input = tf.keras.layers.Input(shape=(7,))  # 7 wide features excluding the 3 categorical features
        
        # Embedding layers for categorical features
        startingAirport_input = tf.keras.layers.Input(shape=(1,))
        startingAirport_embedding = tf.keras.layers.Embedding(self.embedding_sizes['startingAirport'][0], 
                                                            self.embedding_sizes['startingAirport'][1])(startingAirport_input)
        startingAirport_embedding = tf.keras.layers.Flatten()(startingAirport_embedding)
        
        destinationAirport_input = tf.keras.layers.Input(shape=(1,))
        destinationAirport_embedding = tf.keras.layers.Embedding(self.embedding_sizes['destinationAirport'][0], 
                                                        self.embedding_sizes['destinationAirport'][1])(destinationAirport_input)
        destinationAirport_embedding = tf.keras.layers.Flatten()(destinationAirport_embedding)
        
        segmentsCabinCode_input = tf.keras.layers.Input(shape=(1,))
        segmentsCabinCode_embedding = tf.keras.layers.Embedding(self.embedding_sizes['segmentsCabinCode'][0], 
                                                        self.embedding_sizes['segmentsCabinCode'][1])(segmentsCabinCode_input)
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

        self.model = tf.keras.models.Model(inputs=[wide_input, startingAirport_input, destinationAirport_input, segmentsCabinCode_input, deep_input], outputs=output_layer)

    def model_preprocess_data(self):
        train_data, temp_data = train_test_split(self.data, test_size=0.3, random_state=42)
        valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # Split data into wide and deep components for training and validation
        # Convert boolean columns to int
        train_data['flightDate_is_weekend'] = train_data['flightDate_is_weekend'].astype(int)
        valid_data['flightDate_is_weekend'] = valid_data['flightDate_is_weekend'].astype(int)
        test_data['flightDate_is_weekend'] = test_data['flightDate_is_weekend'].astype(int)

        # For the deep component
        train_deep = train_data[['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']].values.astype('float32')
        valid_deep = valid_data[['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']].values.astype('float32')

        # Categorical inputs for wide component
        train_startingAirport = train_data['startingAirport'].values.astype('float32').reshape(-1, 1)
        valid_startingAirport = valid_data['startingAirport'].values.astype('float32').reshape(-1, 1)

        train_destinationAirport = train_data['destinationAirport'].values.astype('float32').reshape(-1, 1)
        valid_destinationAirport = valid_data['destinationAirport'].values.astype('float32').reshape(-1, 1)

        train_segmentsCabinCode = train_data['segmentsCabinCode'].values.astype('float32').reshape(-1, 1)
        valid_segmentsCabinCode = valid_data['segmentsCabinCode'].values.astype('float32').reshape(-1, 1)

        # The other wide features (excluding the 3 categorical features)
        train_other_wide = train_data[['flightDate_year', 'flightDate_month', 'flightDate_day', 'flightDate_weekday', 'flightDate_is_weekend', 'segmentsDepartureTimeRaw_hour', 'segmentsDepartureTimeRaw_minute']].values.astype('float32')
        valid_other_wide = valid_data[['flightDate_year', 'flightDate_month', 'flightDate_day', 'flightDate_weekday', 'flightDate_is_weekend', 'segmentsDepartureTimeRaw_hour', 'segmentsDepartureTimeRaw_minute']].values.astype('float32')

        train_labels = train_data['modeFare'].values.astype('float32')
        valid_labels = valid_data['modeFare'].values.astype('float32')

        return train_data, valid_data, test_data, train_other_wide, train_startingAirport, train_destinationAirport, train_segmentsCabinCode, train_deep, valid_other_wide, valid_startingAirport, valid_destinationAirport, valid_segmentsCabinCode, valid_deep, train_labels, valid_labels

    def compile_model(self, warmup_steps=None):
        initial_learning_rate = 0.01
        
        if warmup_steps:
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[warmup_steps],
                values=[0.005, initial_learning_rate]
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = 'adam'  # Default optimizer if warmup_steps isn't provided
            
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

    def train_model(self, epochs=10, batch_size=32768):
        train_data, self.valid_data, self.test_data, train_other_wide, train_startingAirport, train_destinationAirport, train_segmentsCabinCode, train_deep, valid_other_wide, valid_startingAirport, valid_destinationAirport, valid_segmentsCabinCode, valid_deep, train_labels, valid_labels = self.model_preprocess_data()

        # Determine the warmup steps using the length of train_data
        warmup_epochs = 3
        warmup_steps = warmup_epochs * (len(train_data) // batch_size)
        
        # Compile the model with the determined warmup_steps
        self.compile_model(warmup_steps=warmup_steps)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.005)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint("models/best_model-vishal_raj", save_best_only=True)

        history = self.model.fit(
            [train_other_wide, train_startingAirport, train_destinationAirport, train_segmentsCabinCode, train_deep], 
            train_labels,
            batch_size=batch_size,
            validation_data=([valid_other_wide, valid_startingAirport, valid_destinationAirport, valid_segmentsCabinCode, valid_deep], valid_labels),
            epochs=epochs, 
            callbacks=[early_stop, reduce_lr, model_checkpoint]
        )

        return history

#        tf.keras.models.save_model(self.model, "models/my_final_model")

    def evaluate(self, batch_size=32768):

        # Using the instance variable self.test_data
        test_data = self.test_data
        # Split the test data into wide and deep components
        test_other_wide = test_data[['flightDate_year', 'flightDate_month', 'flightDate_day', 'flightDate_weekday', 'flightDate_is_weekend', 'segmentsDepartureTimeRaw_hour', 'segmentsDepartureTimeRaw_minute']].values.astype('float32')
        test_startingAirport = test_data['startingAirport'].values.astype('float32').reshape(-1, 1)
        test_destinationAirport = test_data['destinationAirport'].values.astype('float32').reshape(-1, 1)
        test_segmentsCabinCode = test_data['segmentsCabinCode'].values.astype('float32').reshape(-1, 1)
        test_deep = test_data[['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']].values.astype('float32')
        
        test_labels = test_data['modeFare'].values.astype('float32')

        # Predictions on the test set
        predictions = self.model.predict([test_other_wide, test_startingAirport, test_destinationAirport, test_segmentsCabinCode, test_deep], batch_size=batch_size)

        # Calculate RMSE and MAE
        rmse = tf.keras.metrics.RootMeanSquaredError()
        mae = tf.keras.metrics.MeanAbsoluteError()

        rmse.update_state(test_labels, predictions)
        mae.update_state(test_labels, predictions)

        return rmse.result().numpy(), mae.result().numpy()
    
    # def hyperparameter_tuning(self, epochs=10):
    #     hypermodel = WideDeepHyperModel(self.embedding_sizes)

    #     tuner = Hyperband(
    #         hypermodel,
    #         objective='val_loss',
    #         max_epochs=epochs,
    #         directory='hyperparam_tuning',
    #         project_name='wide_and_deep'
    #     )

    #     train_data, valid_data, test_data, train_other_wide, train_startingAirport, train_destinationAirport, train_segmentsCabinCode, train_deep, valid_other_wide, valid_startingAirport, valid_destinationAirport, valid_segmentsCabinCode, valid_deep, train_labels, valid_labels = self.model_preprocess_data()

    #     early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    #     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        
    #     tuner.search([train_other_wide, train_startingAirport, train_destinationAirport, train_segmentsCabinCode, train_deep], train_labels, 
    #                  validation_data=([valid_other_wide, valid_startingAirport, valid_destinationAirport, valid_segmentsCabinCode, valid_deep], valid_labels),
    #                  epochs=epochs, 
    #                  callbacks=[early_stop, reduce_lr])

    #     best_model = tuner.get_best_models(num_models=1)[0]
    #     self.model = best_model