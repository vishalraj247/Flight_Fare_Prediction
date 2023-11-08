
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

class xgboost_model():
    def __init__(self):
        self.model = None
        self.val_model = None
        self.best_hyperparameters = None

    
    def get_scores(self, y_true, y_pred, set):
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)

        print(f'SCORES FOR {set}')
        print("-"*50)
        print(f'Root Mean Squared error: {rmse}')
        print(f'Mean Abscolute Error: {mae}')
        print(f'R2 Score: {r2}')

    
    def baseline_model(self, y):
        y_pred = np.full_like(y, y.mean())
        self.get_scores(y, y_pred, 'Baseline Model')


    def train_model(self, X, y, save_model, model_name, file_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
        self.model = model_name
        self.model.fit(X_train, y_train)
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        self.get_scores(y_train, y_pred_train, 'Train Data' + str(model_name))
        self.get_scores(y_test, y_pred_test, 'Test Data' + str(model_name))
        if save_model:
            joblib.dump(self.model, 'models/best_model/' + str(file_name))


    def cross_validate(self, X, y, split):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
        fold_num = 0
        for i in range(split):
            X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, shuffle=True, test_size=0.2, random_state=i)
            self.model = XGBRegressor()
            self.model.fit(X_train_val, y_train_val)
            y_pred_train = self.model.predict(X_train_val)
            y_pred_test = self.model.predict(X_test_val)
            self.get_scores(y_train_val, y_pred_train, 'Train Data for Fold: ' + str(fold_num))
            self.get_scores(y_test_val, y_pred_test, 'Test Data for Fold: ' + str(fold_num))
            fold_num += 1

    def objective(self, params):
        hyperparameters = {
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate'],
            'n_estimators': int(params['n_estimators'])
        }

        model = XGBRegressor(**hyperparameters)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_val)

        mae = mean_absolute_error(self.y_val, y_pred)

        return {'loss': mae, 'status': STATUS_OK}
        

    def tune_xgb(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the hyperparameter search space
        space = {
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
        }

        # Run hyperparameter optimization
        trials = Trials()
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

        self.best_hyperparameters = {
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'n_estimators': int(best['n_estimators'])
        }

        print("Best hyperparameters:", self.best_hyperparameters)


    