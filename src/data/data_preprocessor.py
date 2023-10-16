import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in X.columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
            X[f"{col}_weekday"] = X[col].dt.weekday
            X[f"{col}_is_weekend"] = X[col].dt.weekday >= 5
        return X

# Categorical Feature lists
categorical_features_for_embedding = ['startingAirport', 'destinationAirport', 'segmentsCabinCode']

class DataPreprocessor:
    def __init__(self):
        self.data = None
        self.preprocessor = None
        self.category_mappings = {col: {} for col in categorical_features_for_embedding}
        self.avg_features = pd.DataFrame(columns=['startingAirport', 'destinationAirport', 'median_distance', 'median_duration', 'median_segments_distance'])

    def split_and_explode(self, column):
        self.data = self.data.assign(**{column: self.data[column].str.split('||')}).explode(column)
    
    def preprocess_data(self):
        # Handle columns with '||' entries: split and explode to multiple rows
        columns_to_split_and_explode = [
            'segmentsDepartureTimeRaw', 'segmentsDurationInSeconds',
            'segmentsDistance', 'segmentsCabinCode'
        ]
        for column in columns_to_split_and_explode:
            self.split_and_explode(column)
        
        # Convert segmentsDurationInSeconds and segmentsDistance to numeric
        self.data['segmentsDurationInSeconds'] = pd.to_numeric(self.data['segmentsDurationInSeconds'], errors='coerce')
        self.data['segmentsDistance'] = pd.to_numeric(self.data['segmentsDistance'], errors='coerce')
        
        # Convert segmentsDepartureTimeRaw to datetime
        self.data['segmentsDepartureTimeRaw'] = pd.to_datetime(self.data['segmentsDepartureTimeRaw'], errors='coerce')

        # Now, include segmentsDurationInSeconds and segmentsDistance in numerical_features and segmentsDepartureTimeRaw in datetime_features for further processing
        numerical_features = ['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']
        datetime_features = ['flightDate', 'segmentsDepartureTimeRaw']
        
        # Define transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer_for_embedding = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        datetime_transformer = Pipeline(steps=[
            ('date_features', DateFeatureExtractor()),
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing'))
        ])
        
        # Define the preprocessor after splitting the columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat_emb', categorical_transformer_for_embedding, categorical_features_for_embedding),
                ('date', datetime_transformer, datetime_features)
            ]
        )
        
        # Apply the preprocessor and keep column names
        self.data = pd.DataFrame(self.preprocessor.fit_transform(self.data),
                                 columns=numerical_features + 
                                         [f"{col}_{suffix}" for col in datetime_features for suffix in ['year', 'month', 'day', 'weekday', 'is_weekend']] +
                                         categorical_features_for_embedding)
        
        # Check for any remaining missing values
        self.check_for_missing_values()
        
        return self.data
    
    def check_for_missing_values(self):
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print(f"Warning: There are missing values in the processed data:\n{missing_values[missing_values > 0]}")
        else:
            print("No missing values in the processed data.")
    
    def save_category_mappings(self):
        mappings = {}
        for col in categorical_features_for_embedding:
            unique_values = self.data[col].dropna().unique()
            mappings[col] = {val: i+1 for i, val in enumerate(unique_values)}

        # Add an entry for unknown categories
        for col, mapping in mappings.items():
            mapping['unknown'] = 0

    def preprocess_user_input(self, user_input, preprocessor_path, mappings_path, avg_features_path):
        # Load preprocessor, mappings, and avg_features
        preprocessor = joblib.load(preprocessor_path)
        mappings = joblib.load(mappings_path)
        avg_features = pd.read_csv(avg_features_path)
        
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Map categorical variables using loaded mappings
        for col, mapping in mappings.items():
            input_df[col] = input_df[col].apply(lambda x: x if x in mapping else 'unknown').map(mapping).fillna(0).astype(int)
        
        # Look up average features
        matching_row = avg_features[
            (avg_features['startingAirport'] == user_input['startingAirport']) & 
            (avg_features['destinationAirport'] == user_input['destinationAirport'])
        ]
        
        # Add looked-up average features to input_df
        if not matching_row.empty:
            for col in ['avg_distance', 'avg_duration', 'avg_segments_distance']:
                input_df[col] = matching_row[col].values[0]
        else:
            # Handle scenarios where lookup fails - use median values from the training data
            input_df['avg_distance'] = avg_features['avg_distance'].median()
            input_df['avg_duration'] = avg_features['avg_duration'].median()
            input_df['avg_segments_distance'] = avg_features['avg_segments_distance'].median()
        
        # Apply preprocessor
        preprocessed_input = pd.DataFrame(preprocessor.transform(input_df))
        
        return preprocessed_input

    def process_folder(self, folder):
        folder_path = f'data/interim/{folder}'
        concatenated_data = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.csv')], ignore_index=True)
        self.data = concatenated_data
        processed_data = self.preprocess_data()
        if not os.path.exists(f'data/processed/{folder}'):
            os.makedirs(f'data/processed/{folder}')
        processed_data.to_csv(f'data/processed/{folder}/{folder}_processed.csv', index=False)

    def process_all_folders(self):
        # Process each folder
        for folder in os.listdir('data/interim'):
            if os.path.isdir(f'data/interim/{folder}'):
                self.process_folder(folder)
                # Update category mappings and avg features after processing each folder
                self.update_category_mappings()
                self.update_avg_features_lookup()

        # Save category mappings, avg features, and preprocessor after processing all folders
        self.save_category_mappings()
        self.avg_features.to_csv('../../data/processed/avg_features.csv', index=False)
        joblib.dump(self.preprocessor, '../../models/preprocessor.joblib')

    def save_category_mappings(self):
        for col in categorical_features_for_embedding:
            unique_values = self.data[col].dropna().unique()
            current_mapping = {val: i+1 for i, val in enumerate(unique_values)}
            # Merge with existing mappings
            self.category_mappings[col] = {**self.category_mappings[col], **current_mapping}

        # Add an entry for unknown categories
        for col, mapping in self.category_mappings.items():
            mapping['unknown'] = 0
        
        # Save the category mappings for later use in prediction
        joblib.dump(self.category_mappings, '../../models/category_mappings.joblib')

    def update_avg_features_lookup(self):
        avg_features = self.data.groupby(['startingAirport', 'destinationAirport']).agg({
            'totalTravelDistance': 'median',
            'segmentsDurationInSeconds': 'median',
            'segmentsDistance': 'median'
        }).reset_index()

        avg_features.rename(columns={
            'totalTravelDistance': 'median_distance',
            'segmentsDurationInSeconds': 'median_duration',
            'segmentsDistance': 'median_segments_distance'
        }, inplace=True)

        # Concatenate with existing avg features
        self.avg_features = pd.concat([self.avg_features, avg_features], ignore_index=True).drop_duplicates(subset=['startingAirport', 'destinationAirport'])