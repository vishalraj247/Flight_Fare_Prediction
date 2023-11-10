import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import logging

class RandomForestModel():
    def __init__(self, n_estimators=50, max_depth=10, max_features='sqrt', n_jobs=-1):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=42
        )

    def train_model(self, X, y):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_train)
            self.calculate_rmse(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f'Root Mean Square Error (RMSE): {rmse}')
            
        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")


   
