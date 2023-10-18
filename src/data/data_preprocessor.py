import pandas as pd
import os
import joblib
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Making sure that the input is a DataFrame
        X_copy = pd.DataFrame(X, columns=self.columns_)
        
        date_features_df = pd.DataFrame()
        for col in X_copy.columns:
            if col == "segmentsDepartureTimeRaw":
                # Extract only the hour and minute for segmentsDepartureTimeRaw
                X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')
                date_features_df[f"{col}_hour"] = X_copy[col].dt.hour
                date_features_df[f"{col}_minute"] = X_copy[col].dt.minute
            else:
                # Extract year, month, day, weekday and is_weekend for flightDate
                X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')
                date_features_df[f"{col}_year"] = X_copy[col].dt.year
                date_features_df[f"{col}_month"] = X_copy[col].dt.month
                date_features_df[f"{col}_day"] = X_copy[col].dt.day
                date_features_df[f"{col}_weekday"] = X_copy[col].dt.weekday
                date_features_df[f"{col}_is_weekend"] = X_copy[col].dt.weekday >= 5
        
        return date_features_df

    def fit_transform(self, X, y=None, **fit_params):
        self.columns_ = X.columns
        return self.fit(X).transform(X)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.global_mappings = {}
        for col in columns:
            self.global_mappings[col] = defaultdict(int)
    
    def fit(self, X, y=None):
        print("Fitting the encoder.")
        X_df = pd.DataFrame(X, columns=self.columns)
        for col in self.columns:
            categories = X_df[col].unique()
            print(f"Unique categories in column {col}: {categories}")
            for category in categories:
                if category not in self.global_mappings[col]:
                    self.global_mappings[col][category] = len(self.global_mappings[col])
            print(f"Updated mappings for column {col}: {self.global_mappings[col]}")
        return self
    
    def transform(self, X, y=None):
        print("Transforming the data.")
        X_df = pd.DataFrame(X, columns=self.columns)
        for col in self.columns:
            X_df[col] = X_df[col].map(self.global_mappings[col]).fillna(0).astype(int)
            print(f"Transformed values for column {col}: {X_df[col].unique()}")
        return X_df.values

# Categorical Feature lists
categorical_features_for_embedding = ['startingAirport', 'destinationAirport', 'segmentsCabinCode']

class DataPreprocessor:
    def __init__(self):
        self.data = None
        self.preprocessor = None
        self.category_mappings = {col: {} for col in categorical_features_for_embedding}
        self.avg_features = pd.DataFrame(columns=['median_distance', 'median_duration', 'median_segments_distance'])
        # Instantiate the CategoricalEncoder here
        self.categorical_encoder = CategoricalEncoder(columns=categorical_features_for_embedding)

    def merge_all_datasets(self):
        data_frames = []
        base_path = 'data/interim'
        
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(folder_path, file))
                        data_frames.append(df)
        
        # Concatenate all dataframes into one
        self.data = pd.concat(data_frames, ignore_index=True)

    def split_and_explode(self, columns_to_explode):
        """
        Split and explode columns based on '||' delimiter.
        
        columns_to_explode: List of column names to explode
        """
        # Diagnostic print before explosion
        print("Data before explosion:")
        print(self.data.head())
        # Fill NaN values in segmentsDistance with 'None' before splitting and exploding
        if 'segmentsDistance' in columns_to_explode:
            self.data['segmentsDistance'].fillna('None', inplace=True)

        # Number of splits for each row across the first column
        splits = self.data[columns_to_explode[0]].str.split('\|\|').apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Ensure same number of splits across all columns
        for col in columns_to_explode[1:]:
            col_splits = self.data[col].str.split('\|\|').apply(lambda x: len(x) if isinstance(x, list) else 0)
            if not all(col_splits == splits):
                raise ValueError(f"Columns {columns_to_explode[0]} and {col} do not have the same number of '||' splits.")
        
        # Split and explode
        for col in columns_to_explode:
            self.data[col] = self.data[col].str.split('\|\|')
        
        # Using pandas' explode simultaneously on all columns
        for col in columns_to_explode:
            self.data = self.data.explode(col)

        # Reset the index to ensure unique indices
        self.data.reset_index(drop=True, inplace=True)
        # Diagnostic print after explosion
        print("Data after explosion:")
        print(self.data.head())
        # Check if the column is the datetime column 'segmentsDepartureTimeRaw'
        if 'segmentsDepartureTimeRaw' in columns_to_explode:
            # Ensure that the exploded values are valid datetime strings
            self.data['segmentsDepartureTimeRaw'] = self.data['segmentsDepartureTimeRaw'].str.extract(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})')[0]

    def preprocess_data(self):
        # Handle columns with '||' entries: split and explode to multiple rows
        columns_to_split_and_explode = [
            'segmentsDepartureTimeRaw', 'segmentsDurationInSeconds',
            'segmentsDistance', 'segmentsCabinCode'
        ]
        self.split_and_explode(columns_to_split_and_explode)
        
        # Diagnostic print
        print("Unique values in segmentsCabinCode after split and explode:")
        print(self.data['segmentsCabinCode'].unique())
        # Store the totalFare column after split_and_explode and before applying transformations
        totalFare_column = self.data['totalFare'].copy()

        # Convert segmentsDurationInSeconds and segmentsDistance to numeric
        self.data['segmentsDurationInSeconds'] = pd.to_numeric(self.data['segmentsDurationInSeconds'], errors='coerce')
        self.data['segmentsDistance'] = pd.to_numeric(self.data['segmentsDistance'], errors='coerce')
        
        # Convert segmentsDepartureTimeRaw to datetime
        self.data['segmentsDepartureTimeRaw'] = pd.to_datetime(self.data['segmentsDepartureTimeRaw'], errors='coerce')

        # Define features
        numerical_features = ['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']
        datetime_features = ['flightDate', 'segmentsDepartureTimeRaw']
        
        # Define transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        # Deepcopy the categorical encoder
        encoder_copy = deepcopy(self.categorical_encoder)
        categorical_transformer_for_embedding = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', self.categorical_encoder)
        ])
        datetime_transformer = Pipeline(steps=[
            ('date_features', DateFeatureExtractor())
        ])
        
        # Apply the preprocessor
        if self.preprocessor is None:  # Check if the preprocessor is already instantiated
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat_emb', categorical_transformer_for_embedding, categorical_features_for_embedding),
                    ('date', datetime_transformer, datetime_features)
                ]
            )
            transformed_data = self.preprocessor.fit_transform(self.data)
        else:
            transformed_data = self.preprocessor.transform(self.data)
        
        datetime_extracted_features = [
            f"{col}_{suffix}" for col in datetime_features 
            for suffix in ['year', 'month', 'day', 'weekday', 'is_weekend'] 
            if not (col == "segmentsDepartureTimeRaw" and suffix in ['year', 'month', 'day', 'weekday', 'is_weekend'])
        ] + ["segmentsDepartureTimeRaw_hour", "segmentsDepartureTimeRaw_minute"]
        
        # Combine column names
        all_columns = numerical_features + categorical_features_for_embedding + datetime_extracted_features
        self.data = pd.DataFrame(transformed_data, columns=all_columns)
        
        # Check for any remaining missing values
        self.check_for_missing_values()
        
        # After transforming the data, concatenate the totalFare column back
        self.data['totalFare'] = totalFare_column

        return self.data

    def check_for_missing_values(self):
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print(f"Warning: There are missing values in the processed data:\n{missing_values[missing_values > 0]}")
        else:
            print("No missing values in the processed data.")

    def preprocess_user_input(self, user_input, preprocessor_path, mappings_path, avg_features_path):
        # Load preprocessor, mappings, and avg_features
        preprocessor = joblib.load(preprocessor_path)
        mappings = joblib.load(mappings_path)
        avg_features = pd.read_csv(avg_features_path)
        
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Add a dummy 'totalFare' column
        input_df['totalFare'] = 0
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

    def merge_and_preprocess_all_datasets(self):
        """
        Merge all datasets, preprocess the merged dataset, save the preprocessed data,
        save category mappings, save average features, and the preprocessor.
        """
        # Merge datasets by reading from the airport folders in data/interim
        data_frames = []
        base_path = 'data/interim'
        
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(folder_path, file))
                        data_frames.append(df)
        
        # Concatenate all dataframes into one
        self.data = pd.concat(data_frames, ignore_index=True)
        # Take a random 10% sample of the merged dataset for debugging
        #debug_fraction = 0.1
        #self.data = self.data.sample(frac=debug_fraction).reset_index(drop=True)
        # Preprocess the merged dataset
        processed_data = self.preprocess_data()

        # Save the preprocessed data
        if not os.path.exists(f'data/processed'):
            os.makedirs(f'data/processed')
        processed_data.to_csv(f'data/processed/merged_data_processed.csv', index=False)
        
        # Save category mappings
        self.save_category_mappings()
        joblib.dump(self.category_mappings, 'models/category_mappings.joblib')
        
        # Save average features
        self.save_avg_features_lookup()
        self.avg_features.to_csv('data/processed/avg_features.csv', index=False)
        
        # Save the preprocessor
        joblib.dump(self.preprocessor, 'models/preprocessor.joblib')

    def save_category_mappings(self):
        if self.data is None or self.data.empty:
            print("Warning: self.data is not initialized.")
            return

        for col in categorical_features_for_embedding:
            unique_values = self.data[col].dropna().unique()
            current_mapping = {val: i+1 for i, val in enumerate(unique_values)}
            # Merge with existing mappings
            self.category_mappings[col] = {**self.category_mappings[col], **current_mapping}

        # Add an entry for unknown categories
        for col, mapping in self.category_mappings.items():
            mapping['unknown'] = 0

    def save_avg_features_lookup(self):
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