import os
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost as xgb
import pickle


def save_metrics_excel(dtf_data, metrics, file_path, file_name):
    """
    Funzione per salvare un DataFrame in un file Excel.
    """
    df = dtf_data.copy()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Errore: 'df' deve essere un DataFrame Pandas.")
    if not isinstance(file_path, str):
        raise TypeError("Errore: 'file_path' deve essere una stringa.")
    if not isinstance(file_name, str):
        raise TypeError("Errore: 'file_name' deve essere una stringa.")

    df = df[metrics]
    full_file_path = os.path.join(file_path, file_name)
    df.to_excel(full_file_path, index=False)
    print(f"File salvato con successo in: {full_file_path}")



class Model(object):

    model = None

    """
    class base astratta che definisce il train, predict , save  e caricamento di un modello
    """

    def __init__(self):
        pass

    def train(self, dtf_data, target_col, test_size, random_state):
        pass

    @staticmethod
    def grid_search(dtf_data,target_col,grid_params ,file_path, file_name, estimator=None ,scoring = None, metrics = None):

        if estimator is None:
            raise ValueError("Errore: scegliere un modello per estimatore")
        if scoring is None:
            raise ValueError("Errore: scegliere una loss function per scoring")
        df = dtf_data.copy()

        y = df[target_col]
        x = df.drop(target_col, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        grid_search = GridSearchCV(estimator=estimator, param_grid=grid_params, scoring= scoring, cv=3,
                                 verbose=1, n_jobs=-1, return_train_score=True)
        grid_search.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)
        df_result = pd.DataFrame(grid_search.cv_results_)

        return save_metrics_excel(df_result, metrics, file_path, file_name)

    def predict(self, x):

        if self.model is None:
            raise AttributeError("non è possibile eseguire predict se il modello non è stato addestrato, eseguire prima train")


    def save(self, filepath):
        """
        Serializza e salva l’intero oggetto Model (self) su disco.
        """

        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            raise FileNotFoundError(f"Errore Model.save: la cartella '{dir_name}' non esiste.")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Modello salvato in: {filepath}")

    @staticmethod
    def load(filepath):
        """
        Carica l’intero oggetto Model precedentemente salvato con pickle.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Il file {filepath} non esiste.")
        with open(filepath, "rb") as f:
            models = pickle.load(f)
        print(f"Modello caricato da: {filepath}")
        return models

class XgBoost(Model):
    """
    classe per applicare XgBoost, definita dalla classe base Model, ne eredita tutti i metodi, applica train
    dividendo il df in train e validation (che viene usato per avere una valutazione delle performance)
    predict viene applicato con il modello addestrato in train
    """

    def __init__(self, params):
        super().__init__()
        self.model_params = params

        # if not (isinstance(n_estimators, int) and n_estimators > 0):
        #     raise ValueError("Errore XgBoost: 'n_estimators' deve essere un intero positivo.")
        # if not (isinstance(max_depth, int) and max_depth > 0):
        #     raise ValueError("Errore XgBoost: 'max_depth' deve essere un intero positivo.")
        # if not (isinstance(learning_rate, float) and learning_rate > 0):
        #     raise ValueError("Errore XgBoost: 'learning_rate' deve essere un float > 0.")
        # if not (isinstance(min_child_weight, (int, float)) and min_child_weight > 0):
        #     raise ValueError("Errore XgBoost: 'min_child_weight' deve essere un numero > 0.")
        # if not (isinstance(subsample, float) and 0 < subsample <= 1):
        #     raise ValueError("Errore XgBoost: 'subsample' deve essere un float in (0,1].")
        # if not (isinstance(colsample_bynode, float) and 0 < colsample_bynode <= 1):
        #     raise ValueError("Errore XgBoost: 'colsample_bynode' deve essere un float in (0,1].")
        # if not (isinstance(reg_lambda, float) and reg_lambda >= 0):
        #     raise ValueError("Errore XgBoost: 'reg_lambda' deve essere un float >= 0.")

    @classmethod
    def grid_search(cls,dtf_data,target_col,grid_params ,file_path, file_name, estimator= None,scoring=None, metrics = None):
        df = dtf_data.copy()
        if scoring is None:
            scoring = "neg_root_mean_squared_error"
        if estimator is None:
            estimator = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, early_stopping_rounds=1, enable_categorical= True)
        if metrics is None:
            metrics = df.columns
        return super().grid_search(df, target_col, grid_params, file_path, file_name, estimator=estimator, scoring=scoring, metrics=metrics)


    def train(self, dtf_data, target_col = "cnt", test_size=0.2, random_state=42):
        df = dtf_data.copy()

        self.model = xgb.XGBRegressor(**self.model_params, enable_categorical=True, early_stopping_rounds=10)

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Errore XgBoost.train: 'dtf_data' deve essere un DataFrame Pandas.")
        if target_col not in df.columns:
            raise ValueError(f"Errore XgBoost.train: la colonna target '{target_col}' non è presente nel DataFrame.")
        if not (0 < test_size < 1):
            raise ValueError("Errore XgBoost.train: 'test_size' deve essere un float compreso tra 0 e 1.")
        if not isinstance(random_state, int):
            raise TypeError("Errore XgBoost.train: 'random_state' deve essere un intero.")


        y = df[target_col]
        x = df.drop(target_col, axis=1)

        x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = True)

        return self.model

    def predict(self, x):
        super().predict(x)

        # qua devo mettere il controllo per model, model deve essere non none
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Errore XgBoost.predict: 'x' deve essere un DataFrame Pandas.")
        if x.shape[1] == 0:
            raise ValueError("Errore XgBoost.predict: il DataFrame di input non contiene colonne.")


        return self.model.predict(x)
