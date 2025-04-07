import os
import warnings
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import pickle
from modello.preprocessing import Preprocessing
from pathlib import Path


class Model(ABC):
    """
    Classe base astratta che definisce i metodi fondamentali per il training, la predizione, il fine tuning,
    il salvataggio e il caricamento di un modello.

    Questa classe fornisce una struttura per:
    - Training (addestramento) su dati di input (features + target)
    - Predizione su dati di input tramite il modello addestrato
    - Grid Search (ricerca a griglia) per il tuning degli iperparametri
    - Salvataggio e caricamento dell'intero oggetto modello su disco (tramite pickle)
    """

    model = None

    def __init__(self, test_size=0.2,
                 target_col="cnt",
                 file_path=r"C:\Users\loverdegiulio\Desktop",
                 random_state=42,
                 early_stopping_rounds=10
                 ):
        """
        Inizializza la classe Model.                                                                                

        Parametri                                                                                                   
        ---------                                                                                                   
        test_size : float, default=0.2                                                                              
            Percentuale di dati da destinare al test nella fase di train_test_split (0 < test_size < 1).           
        target_col : str, default="cnt"                                                                             
            Nome della colonna target nel DataFrame.                                                                
        file_path : str, default                                                                                    
            Percorso di default per salvare file Excel o modelli.                                                  
        random_state : int, default=42                                                                              
            Seed per la riproducibilità del train_test_split (deve essere >= 0).                                    
        early_stopping_rounds : int, default=10                                                                     
            Numero di round di early stopping da utilizzare in XGBoost (deve essere > 0).                           
        """  ##
        self.test_size = test_size
        self.target_col = target_col
        self.file_path = file_path
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds

        if not isinstance(test_size, float) or not (0 < test_size < 1):
            raise ValueError(f"Errore {self.__class__.__name__}: 'test_size' deve essere un float compreso tra 0 e 1.")
        if not isinstance(random_state, int) or random_state < 0:
            raise TypeError(f"Errore {self.__class__.__name__}: 'random_state' deve essere un intero non negativo.")
        if not isinstance(target_col, str):
            raise TypeError(f"Errore {self.__class__.__name__}: 'target_col' deve essere una stringa.")
        if not isinstance(file_path, (str, Path)):
            raise TypeError(f"Errore {self.__class__.__name__}: 'file_path' deve essere un percorso in formato "
                            f"stringa.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Errore {self.__class__.__name__}: la cartella '{file_path}' non esiste.")
        if not isinstance(early_stopping_rounds, int) or early_stopping_rounds <= 0:
            raise TypeError(f"Errore {self.__class__.__name__}: 'early_stopping_rounds' deve essere un intero "
                            f"positivo.")

    @staticmethod
    def save_metrics_excel(dtf_data,
                           metrics,
                           file_path,
                           file_name
                           ):
        """
        Salva un DataFrame in un file Excel, filtrando solo le colonne elencate in 'metrics'.

        Parametri
        ---------
        dtf_data : pd.DataFrame
            DataFrame contenente i risultati da salvare.
        metrics : list
            Lista di stringhe con i nomi delle colonne da mantenere nel file Excel.
        file_path : str
            Percorso della cartella di destinazione.
        file_name : str
            Nome del file Excel (es. "risultati.xlsx").

        Returns
        -------
        None
            Non ritorna nulla, ma stampa a video il percorso di salvataggio.
        """
        df = dtf_data.copy()
        df = df[metrics]

        full_file_path = os.path.join(file_path, file_name)
        df.to_excel(full_file_path, index=False)
        print(f"File salvato in: {full_file_path}")

    @abstractmethod
    def _get_model(self,
                   **kwargs
                   ):
        """
        Metodo astratto da implementare nelle sottoclassi per restituire l'oggetto modello.

        Raises
        ------
        NotImplementedError
            Se non implementato in una sottoclasse specifica.
        """
        raise NotImplementedError("Il metodo get_model deve essere implementato nelle sottoclassi.")

    def _split_train_test(self,
                          dtf_data
                          ):

        df = dtf_data.copy()

        y = df[self.target_col]
        x = df.drop(columns=[self.target_col])

        # Train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        return x_train, x_test, y_train, y_test

    def grid_search(self,
                    dtf_data,
                    grid_params,
                    file_name,
                    target_col,
                    early_stopping_rounds=10,
                    validation_size=0.2,
                    scoring=None,
                    metrics=None,
                    **kwargs
                    ):
        """
        Esegue la ricerca a griglia (GridSearchCV) per selezionare gli iperparametri ottimali.

        Parametri
        ---------
        dtf_data : pd.DataFrame
            DataFrame contenente i dati di training (features + target).
        grid_params : dict
            Dizionario con i parametri da testare in GridSearchCV.
        file_name : str
            Nome del file Excel in cui salvare i risultati.
        early_stopping_rounds : int
            Round di early stopping da applicare nei modelli XGBoost.
        cv : int, default=3
            Numero di fold per la validazione incrociata in GridSearchCV.
        scoring : str, default=None
            Metrica di valutazione (es. "neg_mean_squared_error", "r2", ecc.).
        metrics : list, default=None
            Lista delle colonne da salvare nel file Excel; se None, verranno salvate tutte.

        Returns
        -------
        None
            Salva su Excel i risultati della GridSearchCV e non ritorna nulla.
        """
        df = dtf_data.copy()

        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = validation_size
        self.target_col = target_col

        # Controlli basilari sugli argomenti
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Errore {self.__class__.__name__}.grid_search: 'dtf_data' deve essere un DataFrame Pandas."
                            f"")
        if not isinstance(grid_params, dict):
            raise TypeError(f"Errore {self.__class__.__name__}.grid_search: 'grid_params' deve essere un dizionario.")
        if not isinstance(file_name, str):
            raise TypeError(f"Errore {self.__class__.__name__}.grid_search: 'file_name' deve essere una stringa.")
        if scoring is None:
            raise ValueError(f"Errore {self.__class__.__name__}.grid_search: scegliere una loss function per il "
                             f"parametro 'scoring'.")

        x_train, x_val, y_train, y_val = self._split_train_test(df)

        verbose = kwargs.pop("verbose", 1)
        cv = kwargs.pop("cv", 3)
        n_jobs = kwargs.pop("n_jobs", -1)

        grid_search = GridSearchCV(
            estimator=self._get_model(),
            param_grid=grid_params,
            scoring=scoring,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
            return_train_score=True,
            **kwargs)

        grid_search.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True)

        df_result = pd.DataFrame(grid_search.cv_results_)

        if metrics is None:
            metrics = df_result.columns.tolist()
            warnings.warn(
                f"Attenzione {self.__class__.__name__}.grid_search: nessuna metrica di valutazione selezionata,"
                f" verranno restituite tutte le metirche disponibili")
        else:
            invalid_columns = [col for col in metrics if col not in df_result.columns.tolist()]
            if invalid_columns:
                warnings.warn(
                    f"Attenzione: le seguenti colonne non sono valide: {invalid_columns}. "
                    f"Colonne valide: {df_result.columns.tolist()}"
                )

        self.save_metrics_excel(df_result, metrics, self.file_path, file_name)

    def train(self,
              dtf_data
              ):
        """
        Esegue il training del modello utilizzando l'attributo 'model'.

        Parametri
        ---------
        dtf_data : pd.DataFrame
            DataFrame contenente i dati di training (features + target).

        Returns
        -------
        self.model
            Il modello addestrato.

        Raises
        ------
        ValueError
            Se la colonna target non è presente nel DataFrame.
        TypeError
            Se dtf_data non è un DataFrame Pandas.
        """
        df = dtf_data.copy()
        if self.target_col not in dtf_data.columns:
            raise ValueError(f"Errore {self.__class__.__name__}: la colonna target '{self.target_col}'"
                             f" non è presente nel DataFrame.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Errore {self.__class__.__name__}: {dtf_data} deve essere un DataFrame Pandas.")

        x_train, x_test, y_train, y_test = self._split_train_test(df)

        self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)

        return self.model

    def predict(self,
                dtf_pred
                ):
        """
        Effettua la predizione sui dati di input, utilizzando il modello addestrato.

        Parametri
        ---------
        dtf_pred : pd.DataFrame
            DataFrame contenente le feature su cui effettuare la predizione.

        Returns
        -------
        np.array
            Array delle predizioni.

        Raises
        ------
        AttributeError
            Se il modello non è stato addestrato prima di eseguire 'predict'.
        TypeError
            Se dtf_pred non è un DataFrame Pandas.
        ValueError
            Se il DataFrame di input è vuoto.
        """
        if self.model is None:
            raise AttributeError(
                f"Errore {self.__class__.__name__}Non è possibile eseguire 'predict' se il modello non è stato "
                f"addestrato; "
                f"eseguire prima {self.__class__.__name__}.train.")
        if not isinstance(dtf_pred, pd.DataFrame):
            raise TypeError(f"Errore {self.__class__.__name__}: 'x' deve essere un DataFrame Pandas.")
        if dtf_pred.empty:
            raise ValueError(f"Errore {self.__class__.__name__}: il DataFrame di input non contiene colonne.")

        return self.model.predict(dtf_pred)

    def save(self,
             filepath=None
             ):
        """
        Serializza e salva l’intero oggetto Model (self) su disco tramite pickle.

        Parametri
        ---------
        filepath : str, opzionale
            Percorso completo del file dove salvare il modello (es. "modelli/xgboost_model.pkl").
            Se None, viene utilizzato il 'file_path' impostato nel costruttore.

        Returns
        -------
        None
            Non ritorna nulla, ma stampa a video il percorso di salvataggio.

        Raises
        ------
        FileNotFoundError
            Se la cartella di destinazione non esiste.
        """
        if filepath is None:
            filepath = self.file_path

        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            raise FileNotFoundError(f"Errore {self.__class__.__name__}: la cartella '{dir_name}' non esiste.")

        with open(filepath, "wb") as f:
            pickle.dump(self, f)  # type: ignore
        print(f"Modello salvato in: {filepath}")

    @staticmethod
    def load(filepath):
        """
        Carica l'intero oggetto Model precedentemente salvato con pickle.                                           

        Parametri                                                                                                   
        ---------                                                                                                   
        filepath : str                                                                                              
            Percorso completo del file pickle da caricare.                                                          

        Returns                                                                                                     
        -------                                                                                                     
        Model                                                                                                       
            Istanza di Model (o di una sua sottoclasse) caricata da disco.                                          

        Raises                                                                                                      
        ------                                                                                                      
        TypeError                                                                                                   
            Se il filepath non è una stringa o un oggetto Path.                                                     
        FileNotFoundError                                                                                           
            Se il file non esiste nel percorso specificato.                                                         
        """  ##
        if not isinstance(filepath, (str, Path)):
            raise TypeError(
                f"Errore Model.load: filepath' deve essere un percorso in formato stringa.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Il file {filepath} non esiste.")

        with open(filepath, "rb") as f:
            models = pickle.load(f)
        print(f"Modello caricato da: {filepath}")
        return models


class XgBoost(Model):
    """
    Classe per applicare XgBoost, definita a partire dalla classe base Model.
    Ne eredita i metodi fondamentali e implementa la logica di addestramento e predizione con XgBoost.
    """

    def __init__(self,
                 model_parameters=None,
                 test_size=0.2,
                 target_col="cnt",
                 file_path=r"C:\Users\loverdegiulio\Desktop",
                 random_state=42,
                 early_stopping_rounds=10,
                 **kwargs
                 ):
        super().__init__(test_size=test_size,
                         target_col=target_col,
                         file_path=file_path,
                         random_state=random_state,
                         early_stopping_rounds=early_stopping_rounds)
        """
        Inizializza la classe XgBoost.                                                                              

        Parametri                                                                                                   
        ---------                                                                                                   
        model_parameters : dict, opzionale                                                                          
            Dizionario dei parametri da passare a XGBRegressor. Se None, verranno utilizzati i default di XGBRegressor.             
        """  ##

        self.model_params = model_parameters or dict()
        self.additional_params = kwargs

    def _get_model(self, **kwargs):
        """
        Restituisce un regressore XGBRegressor configurato con i parametri specificati.

        Returns
        -------
        xgb.XGBRegressor
            Istanza di regressore XGBoost con i parametri selezionati.
        """
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state,
            enable_categorical=True,
            early_stopping_rounds=self.early_stopping_rounds,
            **kwargs)

    def train(self,
              dtf_data
              ):
        """
        Esegue il training del modello XgBoost, sovrascrivendo se necessario le impostazioni ereditate.

        Parametri
        ---------
        dtf_data : pd.DataFrame
            DataFrame contenente i dati di training (features + target).

        Returns
        -------
        self.model
            Il modello addestrato.
        """
        if not self.model_params:
            warnings.warn("Attenzione: nessun parametro specificato per il modello. Il modello verrà inizializzato con "
                          "i parametri di default Si consiglia di eseguire un fine tuning per ottimizzare le "
                          "performance.")

        self.model = self._get_model(**self.model_params, **self.additional_params)
        return super().train(dtf_data)

