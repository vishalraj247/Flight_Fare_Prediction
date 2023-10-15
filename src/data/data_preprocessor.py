import pandas as pd
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

# Feature lists
numerical_features = ['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']
categorical_features_for_embedding = ['startingAirport', 'destinationAirport', 'segmentsCabinCode']
datetime_features = ['flightDate', 'segmentsDepartureTimeRaw']

# Transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer_for_embedding = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

datetime_transformer = Pipeline(steps=[
    ('date_features', DateFeatureExtractor()),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat_emb', categorical_transformer_for_embedding, categorical_features_for_embedding),
        ('date', datetime_transformer, datetime_features)
    ]
)

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
    
    def preprocess_data(self):
        # Apply the preprocessor
        self.data = pd.DataFrame(preprocessor.fit_transform(self.data))
        
        # Save the preprocessor object for later use in prediction
        joblib.dump(preprocessor, 'path_to_save_preprocessor.joblib')
        
        return self.data
    
    def save_category_mappings(self):
        mappings = {}
        for col in categorical_features_for_embedding:
            unique_values = self.data[col].dropna().unique()
            mappings[col] = {val: i+1 for i, val in enumerate(unique_values)}
        
        # Save the category mappings for later use in prediction
        joblib.dump(mappings, 'path_to_save_category_mappings.joblib')