import os
import sys
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models:dict, hyperparams:dict):
    try:
        report = {}
        for model_name, model in models.items():
            if model_name in hyperparams:

                param_grid = hyperparams[model_name]
                grid_search = GridSearchCV(model, param_grid, scoring='r2', n_jobs=3, cv = 5 , verbose=5)
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_score
            
            else:
                model.fit(X_train, y_train)
                report[model_name] = r2_score(y_test, model.predict(X_test))

        return report


    except Exception as e:
        raise CustomException(e, sys)
    
