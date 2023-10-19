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
import streamlit as st

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

# Categorical Feature lists
categorical_features_for_embedding = ['startingAirport', 'destinationAirport', 'segmentsCabinCode']

class DataPreprocessor:
    def __init__(self):
        self.data = None
        self.category_mappings = {}
        self.preprocessor = None
        self.avg_features = pd.DataFrame()

    def create_category_mappings(self):
        """
        Create mappings for categorical columns.
        """
        for col in categorical_features_for_embedding:
            unique_values = self.data[col].dropna().unique()
            current_mapping = {val: i+1 for i, val in enumerate(unique_values)}
            current_mapping['unknown'] = 0  # Add an entry for unknown categories
            self.category_mappings[col] = current_mapping

    def save_category_mappings(self, path='models/category_mappings.joblib'):
        """
        Save the category mappings to a joblib file.
        """
        if not self.category_mappings:
            print("Warning: No category mappings to save.")
            return
        joblib.dump(self.category_mappings, path)

    def load_category_mappings(self, path='models/category_mappings.joblib'):
        """
        Load the category mappings from a joblib file.
        """
        self.category_mappings = joblib.load(path)

    def encode_categorical_columns(self, df):
        """
        Encode the categorical columns using the saved mappings.
        """
        for col in categorical_features_for_embedding:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.category_mappings[col].get(x, self.category_mappings[col]['unknown']))
        return df

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

        # Map the categorical columns using the category mappings
        self.data = self.encode_categorical_columns(self.data)

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
        categorical_transformer_for_embedding = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
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
        self.load_category_mappings(mappings_path)
        avg_features = pd.read_csv(avg_features_path)
        
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([user_input])
        st.write("Initial Input DataFrame:", input_df)  # DEBUGGING LINE
        
        # Add a dummy 'totalFare' column
        input_df['totalFare'] = 0

        # Display user's input for segmentsCabinCode before mapping
#        st.write(f"User's input for segmentsCabinCode before mapping: {input_df['segmentsCabinCode'].values[0]}")
#        st.write(f"User's input for startingAirport before mapping: {input_df['startingAirport'].values[0]}")
#        st.write(f"User's input for destinationAirport before mapping: {input_df['destinationAirport'].values[0]}")

        # Display the mapping loaded from the joblib file
#        st.write(f"Loaded mappings for segmentsCabinCode: {self.category_mappings['segmentsCabinCode']}")
#        st.write(f"Loaded mappings for startingAirport: {self.category_mappings['startingAirport']}")
#        st.write(f"Loaded mappings for destinationAirport: {self.category_mappings['destinationAirport']}")

        # Map categorical variables using loaded mappings
        for col, mapping in self.category_mappings.items():
            input_df[col] = input_df[col].apply(lambda x: x if x in mapping else 'unknown').map(mapping).fillna(0).astype(int)

        # Display transformed value for segmentsCabinCode after mapping
#        st.write(f"User's input for segmentsCabinCode after mapping: {input_df['segmentsCabinCode'].values[0]}")
#        st.write(f"User's input for startingAirport after mapping: {input_df['startingAirport'].values[0]}")
#        st.write(f"User's input for destinationAirport after mapping: {input_df['destinationAirport'].values[0]}")
#        st.write("After Mapping Categorical Variables:", input_df)  # DEBUGGING LINE
        
        # Look up average features
        matching_row = avg_features[
            (avg_features['startingAirport'] == user_input['startingAirport']) & 
            (avg_features['destinationAirport'] == user_input['destinationAirport'])
        ]

        # Add looked-up average features to input_df
        if not matching_row.empty:
            for avg_col, true_col in zip(['median_distance', 'median_duration', 'median_segments_distance'],
                                        ['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']):
                input_df[true_col] = matching_row[avg_col].values[0]
        else:
            # Handle scenarios where lookup fails - use median values from the training data
            input_df['totalTravelDistance'] = avg_features['median_distance'].median()
            input_df['segmentsDurationInSeconds'] = avg_features['median_duration'].median()
            input_df['segmentsDistance'] = avg_features['median_segments_distance'].median()
        
        st.write("After Adding Average Features:", input_df)  # DEBUGGING LINE

        # Convert the time to string and concatenate with a dummy date
        input_df['segmentsDepartureTimeRaw'] = "2000-01-01 " + input_df['segmentsDepartureTimeRaw'].astype(str)
        input_df['segmentsDepartureTimeRaw'] = pd.to_datetime(input_df['segmentsDepartureTimeRaw'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Get column names after transformation
        datetime_extracted_features = [
            f"{col}_{suffix}" for col in ['flightDate', 'segmentsDepartureTimeRaw']
            for suffix in ['year', 'month', 'day', 'weekday', 'is_weekend']
            if not (col == "segmentsDepartureTimeRaw" and suffix in ['year', 'month', 'day', 'weekday', 'is_weekend'])
        ] + ["segmentsDepartureTimeRaw_hour", "segmentsDepartureTimeRaw_minute"]
        transformed_column_names = ['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance'] + \
                                categorical_features_for_embedding + datetime_extracted_features

        # Apply preprocessor
        preprocessed_input = pd.DataFrame(preprocessor.transform(input_df), columns=transformed_column_names)
        preprocessed_input = preprocessed_input.astype('float32')

        # Overwrite average features
        preprocessed_input.columns = ['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance'] + list(preprocessed_input.columns[3:])
        for col in ['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']:
            preprocessed_input[col] = input_df[col]

        st.write("Final Preprocessed Input:", preprocessed_input)  # DEBUGGING LINE

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
        # Take a random 5% sample of the merged dataset for debugging
        debug_fraction = 0.05
        self.data = self.data.sample(frac=debug_fraction).reset_index(drop=True)

        # Create the category mappings after merging all datasets
        self.create_category_mappings()
        self.save_category_mappings()

        # Preprocess the merged dataset using the created mappings
        processed_data = self.preprocess_data()

        # Save the preprocessed data
        if not os.path.exists(f'data/processed'):
            os.makedirs(f'data/processed')
        processed_data.to_csv(f'data/processed/merged_data_processed.csv', index=False)
        
        # Save average features
        self.save_avg_features_lookup()
        self.avg_features.to_csv('data/processed/avg_features.csv', index=False)
        
        # Save the preprocessor
        joblib.dump(self.preprocessor, 'models/preprocessor.joblib')

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

        # Concatenate with existing avg features, or assign directly if it's empty
        if self.avg_features.empty:
            self.avg_features = avg_features
        else:
            self.avg_features = pd.concat([self.avg_features, avg_features], ignore_index=True).drop_duplicates(subset=['startingAirport', 'destinationAirport'])