import pandas as pd
from modello.preprocessing import Transformation
import bisect


class BinTemp(Transformation):
    """
    Trasforma la colonna 'temp' in categorie basate sulle chiavi di binning.
    """
    def fit(self, dtf_data):
        # Nessuna preparazione necessaria
        pass

    def transform(self, dtf_data):
        df = dtf_data.copy()
        if 'temp' not in df.columns:
            raise ValueError("BinTemp: colonna 'temp' mancante.")
        if not pd.api.types.is_numeric_dtype(df['temp']):
            raise TypeError("BinTemp: 'temp' deve essere numerico.")
        boundaries = [5, 12, 20, 30]
        labels = [0, 1, 2, 3, 4]
        df['temp'] = df['temp'].apply(lambda x: labels[bisect.bisect_left(boundaries, x)])
        return df


class BinAtemp(Transformation):
    """
    Trasforma la colonna 'atemp' in categorie basate sulle chiavi di binning.
    """
    def fit(self, dtf_data):
        pass

    def transform(self, dtf_data):
        df = dtf_data.copy()
        if 'atemp' not in df.columns:
            raise ValueError("BinAtemp: colonna 'atemp' mancante.")
        if not pd.api.types.is_numeric_dtype(df['atemp']):
            raise TypeError("BinAtemp: 'atemp' deve essere numerico.")
        boundaries = [5, 12, 22, 32, 40]
        labels = [0, 1, 2, 3, 4, 5]
        df['atemp'] = df['atemp'].apply(lambda x: labels[bisect.bisect_left(boundaries, x)])
        return df


class BinHum(Transformation):
    """
    Trasforma la colonna 'hum' in categorie basate sulle chiavi di binning.
    """
    def fit(self, dtf_data):
        pass

    def transform(self, dtf_data):
        df = dtf_data.copy()
        if 'hum' not in df.columns:
            raise ValueError("BinHum: colonna 'hum' mancante.")
        if not pd.api.types.is_numeric_dtype(df['hum']):
            raise TypeError("BinHum: 'hum' deve essere numerico.")
        boundaries = [30, 45, 60, 80]
        labels = [0, 1, 2, 3, 4]
        df['hum'] = df['hum'].apply(lambda x: labels[bisect.bisect_left(boundaries, x)])
        return df


class BinWind(Transformation):
    """
    Trasforma la colonna 'windspeed' in categorie basate sulle chiavi di binning.
    """
    def fit(self, dtf_data):
        pass

    def transform(self, dtf_data):
        df = dtf_data.copy()
        if 'windspeed' not in df.columns:
            raise ValueError("BinWind: colonna 'windspeed' mancante.")
        if not pd.api.types.is_numeric_dtype(df['windspeed']):
            raise TypeError("BinWind: 'windspeed' deve essere numerico.")
        boundaries = [10, 20, 30]
        labels = [1, 2, 3, 4]
        df['windspeed'] = df['windspeed'].apply(lambda x: labels[bisect.bisect_left(boundaries, x)])
        return df


class Postprocessing:
    """
    Applica tutte le trasformazioni di post-processing in sequenza.
    """
    def __init__(self):
        self.dct_trans = {
            'temp': BinTemp(),
            'atemp': BinAtemp(),
            'hum': BinHum(),
            'windspeed': BinWind()
        }

    def transform(self, dtf_data):
        if not isinstance(dtf_data, pd.DataFrame):
            raise TypeError("Postprocessing: input non è un DataFrame Pandas.")
        df = dtf_data.copy()
        for name, trans in self.dct_trans.items():
            trans.fit(df)
            df = trans.transform(df)
        return df

if __name__ == "__main__":
    from preprocessing import  Preprocessing
    from sklearn.model_selection import train_test_split




    df_test = pd.read_csv("datas/hour.csv")

    df_train, df_predict = train_test_split(df_test, test_size=0.2, random_state=42)

    preproc = Preprocessing()

    df_train_encod = preproc.transform_data(df_train, "train")
    df_test_encod = preproc.transform_data(df_predict, "predict")

    df_test = df_test_encod.copy()
    df_test_encod.drop(columns=["cnt"], inplace=True)

    postproc = Postprocessing()
    df_test_post = postproc.transform(df_test)
