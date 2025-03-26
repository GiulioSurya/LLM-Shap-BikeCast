import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost as xgb
import bisect
import pickle
from preprocessing import Preprocessing


class Model(object):
    model = None
    test_size = 0.2
    target_col = "cnt"
    file_path = r"C:\Users\loverdegiulio\Desktop"
    random_state = 42

    """
    class base astratta che definisce il train, predict , save  e caricamento di un modello
    """

    def __init__(self):
        pass

    @staticmethod
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
        return print (f"File salvato in: {full_file_path}")

    @classmethod
    def get_model(cls):
        return  NotImplementedError("Il metodo get_model deve essere implementato nelle sottoclassi.")

    @classmethod
    def grid_search(cls,dtf_data,grid_params, file_name, target_col = target_col, file_path = file_path , scoring = None, metrics = None):
        estimator = cls.get_model()

        if scoring is None:
            raise ValueError("Errore: scegliere una loss function per scoring")

        df = dtf_data.copy()
        y = df[target_col]
        x = df.drop(target_col, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cls.test_size, random_state=42)

        grid_search = GridSearchCV(estimator=estimator, param_grid=grid_params, scoring= scoring, cv=3,
                                 verbose=1, n_jobs=-1, return_train_score=True)
        grid_search.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)
        df_result = pd.DataFrame(grid_search.cv_results_)

        if metrics is None:
            metrics = df_result.columns.tolist()
            warnings.warn("nessuna metrica selezionata, verranno restituite tutte le metriche disponibili, usare il parametro metrics per selezionare le colonne desiderate.")

        return cls.save_metrics_excel(df_result, metrics, file_path, file_name)

    def train(self, dtf_data, target_col = target_col, test_size = test_size, random_state = random_state):
        df = dtf_data.copy()


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

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)

        return self.model

    def predict(self, x):

        if self.model is None:
            raise AttributeError("non è possibile eseguire predict se il modello non è stato addestrato, eseguire prima train")
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Errore Model.predict: 'x' deve essere un DataFrame Pandas.")
        if x.shape[1] == 0:
            raise ValueError("Errore Model.predict: il DataFrame di input non contiene colonne")

        return self.model.predict(x)


    def save(self, filepath = file_path):
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
    def load(filepath = file_path):
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

    @classmethod
    def get_model(cls, **kwargs):
        return xgb.XGBRegressor(objective='reg:squarederror', random_state=42, early_stopping_rounds=10, enable_categorical= True, **kwargs)


    def train(self, dtf_data, target_col=None, test_size=None, random_state=None):

        if target_col is None:
            target_col = self.target_col
        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state

        self.model = self.get_model(**self.model_params)

        return super().train(dtf_data, target_col=target_col, test_size=test_size, random_state=random_state)

