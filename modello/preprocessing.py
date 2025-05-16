import os
import pandas as pd
import bisect
import pickle



class Transformation(object):
    """
    Classe base per le trasformazioni dei dati.
    Definisce un'interfaccia con metodi 'fit' e 'transform'
    che possono essere implementati dalle sottoclassi.
    """

    def __init__(self):
        pass

    def fit(self, dtf_data):
        """
        Metodo 'fit' di base, da ridefinire nelle sottoclassi se necessario.
        """
        pass

    def transform(self, dtf_data):
        """
        Metodo 'transform' di base, da implementare nelle sottoclassi.
        """
        pass

class DummyEncoderForHours:
    def __init__(self):
        self.dict_mean_dummies = None

    def _bin_transformer(self, hr: int) -> str:
        """
        Trasforma l'ora in fascia oraria (es. 'morning', 'afternoon'...).
        Personalizza questo metodo come preferisci.
        """
        if hr < 6:
            return "night"
        elif hr < 12:
            return "morning"
        elif hr < 18:
            return "afternoon"
        else:
            return "evening"

    def _bin_value(self, dtf_data: pd.DataFrame, require_target: bool = True) -> pd.Series:
        """
        Crea una variabile categoriale unendo la fascia oraria e il working day.

        Parametri
        ---------
        dtf_data : pd.DataFrame
            Dataset contenente le colonne 'hr' e 'workingday'.
        require_target : bool, default=True
            Se True, richiede la presenza della colonna 'cnt' per validazione.

        Returns
        -------
        pd.Series
            Serie con etichette del tipo 'fascia_oraria_workingday'.
        """
        df = dtf_data.copy()

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Errore DummyEncoderForHours: 'dtf_data' deve essere un DataFrame Pandas.")
        if "hr" not in df.columns:
            raise ValueError("Errore DummyEncoderForHours: non è presente la colonna 'hr' nei dati.")
        if "workingday" not in df.columns:
            raise ValueError("Errore DummyEncoderForHours: non è presente la colonna 'workingday' nei dati.")
        if require_target and "cnt" not in df.columns:
            raise ValueError("Errore DummyEncoderForHours: non è presente la colonna target 'cnt' nei dati.")

        if not pd.api.types.is_numeric_dtype(df["hr"]):
            raise TypeError("Errore DummyEncoderForHours: la colonna 'hr' deve essere di tipo numerico.")
        if not pd.api.types.is_numeric_dtype(df["workingday"]):
            raise TypeError("Errore DummyEncoderForHours: la colonna 'workingday' deve essere di tipo numerico.")

        df["time_bin"] = df["hr"].apply(self._bin_transformer)

        return df["time_bin"].astype(str) + "_" + df["workingday"].astype(str)

    def fit(self, dtf_data: pd.DataFrame) -> None:
        """
        Calcola i valori medi del target 'cnt' per le diverse combinazioni
        fascia oraria-working day.
        """
        df = dtf_data.copy()
        hr_wd = self._bin_value(df, require_target=True)

        if "cnt" not in df.columns:
            raise ValueError("Errore DummyEncoderForHours: manca la colonna 'cnt'.")
        if not pd.api.types.is_numeric_dtype(df["cnt"]):
            raise TypeError("Errore DummyEncoderForHours: la colonna 'cnt' deve essere di tipo numerico.")

        self.dict_mean_dummies = df.groupby(hr_wd)["cnt"].mean().to_dict()

    def transform(self, dtf_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge la colonna 'hw_wd_cnt_enc' mappando i valori medi calcolati.
        """
        if self.dict_mean_dummies is None:
            raise ValueError("Errore DummyEncoderForHours: 'fit' non è stato chiamato prima di 'transform'.")

        df = dtf_data.copy()
        hr_wd = self._bin_value(df, require_target=False)
        df["hw_wd_cnt_enc"] = hr_wd.map(self.dict_mean_dummies)

        return df



class WindToCategorical(Transformation):
    """
    Classe che trasforma la velocità del vento in categorie discrete.
    """

    @staticmethod
    def _bin_wind(wind):
        """
        Trasforma il valore di wind in uno dei 4 intervalli definiti.
        """
        if wind is (wind < 0):
            raise ValueError("Attenzione, wind deve essere un numero positivo")

        boundaries = [10, 20, 30]
        labels = [1, 2, 3, 4]
        #labels = ["0-10", "11-20", "21-30", ">30"]

        return labels[bisect.bisect_left(boundaries, wind)]

    def transform(self, dtf_data):
        """
        Scala la velocità del vento, poi la converte in categorie.
        """
        df = dtf_data.copy()

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Errore WindToCategorical: 'dtf_data' deve essere un DataFrame Pandas.")
        if "windspeed" not in df.columns:
            raise ValueError("Errore WindToCategorical: non è presente la colonna 'windspeed'.")
        if not pd.api.types.is_numeric_dtype(df["windspeed"]):
            raise TypeError("Errore WindToCategorical: 'windspeed' deve essere di tipo numerico.")

        df["windspeed"] = df["windspeed"] * 41
        df["windspeed"] = df["windspeed"].apply(self._bin_wind)

        return df


class TempTransformation(Transformation):
    """
    Classe che scala il valore di 'temp' moltiplicandolo per 41.
    """

    def transform(self, dtf_data):
        """
        Applica il fattore di scala 41 alla colonna 'temp'.
        """
        df = dtf_data.copy()

        if "temp" not in df.columns:
            raise ValueError("Errore TempTransformation: non è presente la colonna 'temp'.")
        if not pd.api.types.is_numeric_dtype(df["temp"]):
            raise TypeError("Errore TempTransformation: 'temp' deve essere numerico.")

        df["temp"] = df["temp"] * 41

        return df


class AtempTransformation(Transformation):
    """
    Classe che scala il valore di 'atemp' moltiplicandolo per 50.
    """

    def transform(self, dtf_data):
        """
        Applica il fattore di scala 50 alla colonna 'atemp'.
        """
        df = dtf_data.copy()

        if "atemp" not in df.columns:
            raise ValueError("Errore AtempTransformation: non è presente la colonna 'atemp'.")
        if not pd.api.types.is_numeric_dtype(df["atemp"]):
            raise TypeError("Errore AtempTransformation: 'atemp' deve essere numerico.")

        df["atemp"] = df["atemp"] * 50

        return df


class HumTransformation(Transformation):
    """
    Classe che scala il valore di 'hum' e rimuove i record
    con valori inferiori ad una soglia minima.
    """

    def transform(self, dtf_data):
        """
        Moltiplica la colonna 'hum' per 100 e
        rimuove i valori minori di 17.
        """
        df = dtf_data.copy()

        if "hum" not in df.columns:
            raise ValueError("Errore HumTransformation: non è presente la colonna 'hum'.")
        if not pd.api.types.is_numeric_dtype(df["hum"]):
            raise TypeError("Errore HumTransformation: 'hum' deve essere numerico.")

        df["hum"] = df["hum"] * 100
        df.drop(index=df[df["hum"] < 17].index, inplace=True)

        return df


class WeekNewVar(Transformation):
    """
    Classe che aggiunge la variabile 'weekofyear'
    in base alla data presente nella colonna 'dteday'.
    """

    def transform(self, dtf_data):
        """
        Converte 'dteday' in datetime e aggiunge la colonna 'weekofyear'.
        """
        df = dtf_data

        if "dteday" not in df.columns:
            raise ValueError("Errore WeekNewVar: non è presente la colonna 'dteday' nei dati.")
        try:
            df["dteday"] = pd.to_datetime(df["dteday"])
        except Exception as e:
            raise ValueError(f"Errore WeekNewVar: impossibile convertire 'dteday' in datetime. Dettagli: {e}")

        df["weekofyear"] = df["dteday"].dt.isocalendar().week.astype(int)

        return df


class DropVariable(Transformation):
    """
    Classe che elimina dal DataFrame un set predefinito di colonne non utilizzate.
    """

    def transform(self, dtf_data):
        """
        Elimina le colonne specificate se presenti nel DataFrame.
        """
        df = dtf_data.copy()
        colonne_da_rimuovere = ["casual", "registered", "mnth", "instant", "season", "yr", "dteday"]

        # Elimina solo le colonne presenti
        return df.drop(columns=colonne_da_rimuovere, errors='ignore')


class ChangeType(Transformation):
    """
    Classe che converte alcune colonne in categorie e altre in boolean.
    """

    def transform(self, dtf_data):
        df = dtf_data.copy()
        conversions = {
            "weathersit": int,
            "weekofyear": int,
            "windspeed": int,
            "weekday": int,
            "holiday": bool,
            "workingday": bool
        }

        for col, new_type in conversions.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(new_type)
                    # Converti bool in int (True/False → 1/0)
                    if new_type == bool:
                        df[col] = df[col].astype(int)
                except ValueError as e:
                    raise TypeError(
                        f"Errore ChangeType: impossibile convertire la colonna '{col}' in {new_type}.") from e

        return df


class Preprocessing(object):
    """
    Classe che gestisce la pipeline di trasformazione su un DataFrame
    attraverso un dizionario di trasformazioni definite.
    """

    def __init__(self):
        """
        Inizializza il dizionario di trasformazioni, con chiavi descrittive
        e istanze delle classi di trasformazione corrispondenti.
        """
        self.dct_trans = {
            "temp": TempTransformation(),
            "dummy_ecod": DummyEncoderForHours(),
            "wind_cathegorical": WindToCategorical(),
            "new_var_weekofyear": WeekNewVar(),
            "atemp": AtempTransformation(),
            "hum": HumTransformation(),
            "drop_variable": DropVariable(),
            "change_type": ChangeType()
        }

    def transform_data(self, dtf_data, str_method="train"):
        """
        Applica la pipeline di trasformazioni:
        - in 'train' mode, viene chiamato 'fit' per ogni trasformazione (se definito),
          poi 'transform'.
        - in 'predict' mode, viene chiamato solo 'transform'.
        """

        if not isinstance(dtf_data, pd.DataFrame):
            raise TypeError("Errore Preprocessing: 'dtf_data' deve essere un DataFrame Pandas.")
        if str_method not in ["train", "predict"]:
            raise ValueError("Errore Preprocessing: 'str_method' deve essere 'train' o 'predict'.")

        if str_method == "train":
            for _, trans in self.dct_trans.items():
                trans.fit(dtf_data)
                dtf_data = trans.transform(dtf_data)
        if str_method == "predict":
            for _, trans in self.dct_trans.items():
                dtf_data = trans.transform(dtf_data)

        return dtf_data

    def save(self, filepath):
        """
        Serializza l'intera pipeline di Preprocessing su disco.
        """
        # **AGGIUNTO**: controllo cartella esistente
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            raise FileNotFoundError(f"Errore Preprocessing.save: la cartella '{dir_name}' non esiste.")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Preprocessing salvato in: {filepath}")

    @staticmethod
    def load(filepath):
        """
        Carica da disco un oggetto Preprocessing precedentemente serializzato.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Errore Preprocessing.load: il file '{filepath}' non esiste.")
        with open(filepath, "rb") as f:
            preprocedure = pickle.load(f)
        print(f"Preprocessing caricato da: {filepath}")
        return preprocedure
