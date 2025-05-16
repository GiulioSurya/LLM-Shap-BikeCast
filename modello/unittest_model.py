import unittest
from unittest.mock import Mock, patch, mock_open
import os
import pandas as pd
from modello import XgBoost, Model


# ------------ TEST INPUT

class TestModel(unittest.TestCase):
    """
    Test per validare la corretta gestione dei parametri in ingresso
    di Model e della sua sottoclasse XgBoost.
    """

    # =========================
    # 1) Test costruttore
    # =========================
    def test_init_valid_input(self):
        """
        Verifica che i parametri validi non sollevino eccezioni
        e che le variabili d'istanza abbiano il tipo corretto.
        """
        # Esempio identico al tuo snippet
        test = XgBoost(
            test_size=0.4,
            target_col="test",
            file_path=r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello",
            early_stopping_rounds=5
        )
        self.assertIsInstance(test.test_size, float)
        self.assertIsInstance(test.target_col, str)
        self.assertIsInstance(test.file_path, str)
        self.assertIsInstance(test.early_stopping_rounds, int)

    def test_init_test_size_non_float(self):
        """
        Verifica che venga sollevata un'eccezione se 'test_size' non è float.
        """
        with self.assertRaises(ValueError):
            XgBoost(
                test_size="0.2"
            )

    def test_init_test_size_out_of_range(self):
        """
        Verifica che venga sollevata un'eccezione se 'test_size' non è (0,1).
        """
        with self.assertRaises(ValueError):
            XgBoost(
                test_size=1.5
            )

    def test_init_random_state_negative(self):
        """
        Verifica che venga sollevata TypeError se 'random_state' < 0.
        """
        with self.assertRaises(TypeError):
            XgBoost(
                random_state=-1,
            )

    def test_init_target_col_not_str(self):
        """
        Verifica che venga sollevata TypeError se 'target_col' non è stringa.
        """
        with self.assertRaises(TypeError):
            XgBoost(
                target_col=1234,
            )

    def test_init_file_path_not_str_nor_path(self):
        """
        Verifica che venga sollevata TypeError se 'file_path' non è str né Path.
        """
        with self.assertRaises(TypeError):
            XgBoost(
                file_path=1234
            )

    def test_init_file_path_not_exist(self):
        """
        Verifica che venga sollevata FileNotFoundError se la cartella non esiste.
        """
        with self.assertRaises(FileNotFoundError):
            XgBoost(
                file_path=r"C:\percorso\inesistente\di\sicuro"
            )

    def test_init_early_stopping_non_positive(self):
        """
        Verifica che venga sollevata TypeError se 'early_stopping_rounds' <= 0.
        """
        with self.assertRaises(TypeError):
            XgBoost(
                early_stopping_rounds=-5
            )
    def test_init_early_stopping_rounds_string(self):
        """
        Verifica che venga sollevata TypeError se 'early_stopping_rounds' non è int.
        """
        with self.assertRaises(TypeError):
            XgBoost(
                early_stopping_rounds="test"
            )

    # =========================
    # 2) Test grid_search
    # =========================
    def test_grid_search_not_dataframe(self):
        """
        Verifica che grid_search sollevi TypeError se dtf_data non è un DataFrame.
        """
        with self.assertRaises(TypeError):
            XgBoost().grid_search(
                dtf_data=[1, 2, 3],
                grid_params={},
                file_name="output.xlsx",
                target_col="cnt",
                scoring="neg_mean_squared_error"
            )

    def test_grid_search_grid_params_not_dict(self):
        """
        Verifica che grid_search sollevi TypeError se grid_params non è un dizionario.
        """
        df = pd.DataFrame({"cnt": [1, 2], "feature": [0.1, 0.2]})
        with self.assertRaises(TypeError):
            XgBoost().grid_search(
                dtf_data=df,
                grid_params= {},
                file_name="output.xlsx",
                target_col="cnt",
                scoring="neg_mean_squared_error"
            )

    def test_grid_search_file_name_not_str(self):
        """
        Verifica che grid_search sollevi TypeError se file_name non è una stringa.
        """
        df = pd.DataFrame({"cnt": [1, 2], "feature": [0.1, 0.2]})
        with self.assertRaises(TypeError):
            XgBoost().grid_search(
                dtf_data=df,
                grid_params={"n_estimators": [100]},
                file_name=123,
                target_col="cnt",
                scoring="neg_mean_squared_error"
            )

    def test_grid_search_scoring_none(self):
        """
        Verifica che grid_search sollevi ValueError se scoring è None.
        """
        model = XgBoost()
        df = pd.DataFrame({"cnt": [1, 2], "feature": [0.1, 0.2]})
        with self.assertRaises(ValueError):
            model.grid_search(
                dtf_data=df,
                grid_params={"n_estimators": [100]},
                file_name="test.xlsx",
                target_col="cnt",
                scoring=None
            )

    # =========================
    # 3) Test train & predict
    # =========================
    def test_train_missing_target(self):
        """
        Verifica che venga sollevato ValueError se la colonna target non è presente.
        """
        model = XgBoost()
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        with self.assertRaises(ValueError):
            model.train(df)

    def test_predict_before_train(self):
        """
        Verifica che predict sollevi AttributeError se il modello non è stato allenato.
        """
        model = XgBoost()
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with self.assertRaises(AttributeError):
            model.predict(df)

    def test_predict_input_not_df(self):
        """
        Verifica che predict sollevi TypeError se dtf_pred non è un DataFrame.
        """
        model = XgBoost()
        # Alleniamo con un DataFrame minimo
        df = pd.DataFrame({"cnt": [1, 2], "feature": [1, 2]})
        model.train(df)
        with self.assertRaises(TypeError):
            model.predict([1, 2, 3])

    def test_predict_empty_df(self):
        """
        Verifica che predict sollevi ValueError se il DataFrame è vuoto.
        """
        model = XgBoost()
        df = pd.DataFrame({"cnt": [1, 2], "feature": [1, 2]})
        model.train(df)
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            model.predict(empty_df)

    # =========================
    # 4) Test save & load
    # =========================
    def test_save_invalid_directory(self):
        """
        Verifica che venga sollevato FileNotFoundError se la cartella di destinazione non esiste.
        """
        model = XgBoost()
        df = pd.DataFrame({"cnt": [1, 2], "feature": [1, 2]})
        model.train(df)

        invalid_path = r"C:\percorso\inesistente\test_model.pkl"
        with self.assertRaises(FileNotFoundError):
            model.save(invalid_path)

    def test_load_invalid_path_type(self):
        """
        Verifica che venga sollevato TypeError se il filepath di load non è stringa né Path.
        """
        with self.assertRaises(TypeError):
            Model.load(123)

    def test_load_file_not_exists(self):
        """
        Verifica che venga sollevato FileNotFoundError se il file di destinazione non esiste.
        """
        invalid_file = r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello\inesistente.pkl"
        with self.assertRaises(FileNotFoundError):
            Model.load(invalid_file)

    def test_save_and_load_ok(self):
        """
        Verifica che il salvataggio e il caricamento funzionino senza sollevare eccezioni
        e che il modello caricato sia utilizzabile.
        """
        # Creiamo un file di destinazione in una cartella esistente
        valid_folder = r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello"
        save_path = os.path.join(valid_folder, "xgb_test_model.pkl")

        # Alleniamo un modello minimale
        model = XgBoost(file_path=valid_folder)
        df = pd.DataFrame({"cnt": [1, 2], "feature": [1, 2]})
        model.train(df)

        # Salvataggio
        model.save(save_path)

        # Caricamento
        loaded_model = Model.load(save_path)
        self.assertIsNotNone(loaded_model.model, "Il modello caricato non deve essere None.")
        self.assertIsInstance(loaded_model, XgBoost, "Dopo il load, l'oggetto deve essere di tipo XgBoost.")

        # Proviamo una predizione per ulteriore conferma
        pred_df = pd.DataFrame({"feature": [3, 4]})
        preds = loaded_model.predict(pred_df)
        self.assertEqual(len(preds), 2, "Le predizioni devono avere la stessa lunghezza dei dati di input.")


# ---------------- TEST OUTPUT and CORPUS
class ModelForTest(Model):
    def _get_model(self,
                   **kwargs
                   ):
        return None

class TestOutputModel(unittest.TestCase):

    def setUp(self):
        self.dtf_data = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "cnt": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        })

        self.dtf_pred = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "cnt": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        })

        self.target = "cnt"
        self.file_path = os.getcwd()
        self.file_name = "test_output.xlsx"



    @patch("pandas.DataFrame.to_excel")
    @patch("builtins.print")
    def test_output_save_metrics_excel(self, mock_print, mock_to_excel):

        metrics = ["feature"]
        file_path = r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello"
        file_name = "test_output.xlsx"

        expected_full_path = os.path.join(file_path, file_name)

        XgBoost.save_metrics_excel(self.dtf_data, metrics, file_path, file_name)

        mock_print.assert_called_once_with(f"File salvato in: {expected_full_path}")

        mock_to_excel.assert_called_once_with(expected_full_path, index=False)

    def test_output_get_model_abc(self):
        test = Model()
        with self.assertRaises(NotImplementedError) as context:
            test._get_model()
        self.assertIn("Il metodo get_model deve essere implementato nelle sottoclassi.", str(context.exception))


    def test_split_train_output_lenght(self):

        model = ModelForTest()

        x_train, x_test, y_train, y_test = model._split_train_test(self.dtf_data)

        self.assertEqual(len(x_train), 8)
        self.assertEqual(len(x_test), 2)
        self.assertEqual(len(y_train), 8)
        self.assertEqual(len(y_test), 2)

    def test_split_train_output_target(self):

        model = ModelForTest()
        x_train, x_test, y_train, y_test = model._split_train_test(self.dtf_data)

        self.assertNotIn(self.target, x_train.columns)
        self.assertNotIn(self.target, x_test.columns)

    @patch("modello.GridSearchCV")
    @patch("pandas.DataFrame.to_excel")
    @patch("builtins.print")
    def test_grid_search_fit_and_save_called(self, mock_print, mock_excel, mock_gridsearch):
        # 1. Preparo il mock che verrà restituito quando XgBoost chiama GridSearchCV(...)
        mock_grid = Mock()
        mock_grid.fit.return_value = mock_grid  # .fit() deve restituire sé stesso
        mock_grid.cv_results_ = {
            'mean_test_score': [0.1],
            'std_test_score': [0.01],
            'params': [{'n_estimators': 100}]
        }

        # 2. Dico al patch di restituire il mio mock quando viene chiamato GridSearchCV(...)
        mock_gridsearch.return_value = mock_grid

        # 3. Istanzio il modello e chiamo il metodo da testare
        model = XgBoost(file_path=self.file_path)
        model.grid_search(
            dtf_data=self.dtf_data,
            grid_params={"n_estimators": [100]},
            file_name=self.file_name,
            target_col=self.target,
            scoring="neg_mean_squared_error"
        )

        # 4. Verifico che il file Excel sia stato salvato
        full_file_path = os.path.join(self.file_path, self.file_name)
        mock_print.assert_called_once_with(f"File salvato in: {full_file_path}")
        mock_excel.assert_called_once_with(full_file_path, index=False)

        # 5. Verifico che il fit sia stato chiamato
        mock_grid.fit.assert_called_once()

    @patch("modello.GridSearchCV")
    def test_grid_search_metrics_none(self, mock_grid):
        mock_pers = Mock()
        mock_pers.fit.return_value = mock_pers
        mock_pers.cv_results_ = mock_grid.cv_results_ = {
            "mean_test_score": [0.5],
            "std_test_score": [0.01],
            "params": [{"n_estimators": 100}]}

        mock_grid.return_value = mock_pers

        model = XgBoost()

        with self.assertWarns(UserWarning) as warn:
            model.grid_search(
                dtf_data=self.dtf_data,
                grid_params={"n_estimators": [100]},
                file_name=self.file_name,
                target_col=self.target,
                scoring="neg_mean_squared_error")

        self.assertIn("nessuna metrica di valutazione selezionata,", str(warn.warning))

    @patch("modello.Model.save_metrics_excel")
    @patch("modello.GridSearchCV")
    @patch("pandas.Index.tolist")
    def test_grid_search_invalid_col(self, mock_tolist ,mock_grid, mock_save_excel):
        mock_pers = Mock()
        mock_pers.fit.return_value = mock_pers
        mock_pers.cv_results_ = mock_grid.cv_results_ = {
            "mean_test_score": [0.5],
            "std_test_score": [0.01],
            "params": [{"n_estimators": 100}]}

        mock_grid.return_value = mock_pers
        #allora fino a mo ho evitato il grid serach, ma quello che voglio è patchare
        #tot list, cosi so quali colonne ci sono dentro

        mock_tolist.return_value = ["colonna1", "colonna2", "colonna3"]

        mock_save_excel.return_value = None

        #ora posso istanziare tutto e metric metterò colonna 4

        model = XgBoost()
        with self.assertWarns(UserWarning) as context:
            model.grid_search(
                dtf_data=self.dtf_data,
                grid_params={"n_estimators": [100]},
                file_name=self.file_name,
                target_col=self.target,
                scoring="neg_mean_squared_error",
                metrics=["colonna4"]
            )
        self.assertIn("Attenzione: le seguenti colonne non sono valide", str(context.warning))

    @patch("modello.Model._split_train_test")
    @patch("modello.XgBoost._get_model")
    def test_train_return_model(self, mock_get_model, mock_split):

        mock_split.return_value = (
            pd.DataFrame({"feature": [1, 2, 3]}),
            pd.DataFrame({"feature": [4, 5]}),
            pd.Series([1, 2, 3]),
            pd.Series([4, 5])
        )

        get_mock = Mock()
        get_mock.fit.return_value = get_mock

        mock_get_model.return_value = get_mock

        model_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "early_stopping_rounds": 5
        }

        model = XgBoost(model_params)

        result = model.train(self.dtf_data)

        self.assertEqual(result, get_mock)

    @patch("modello.Model._split_train_test")
    @patch("modello.XgBoost._get_model")
    def test_train_assert_warning(self, mock_get_model, mock_split):

        mock_split.return_value = (
            pd.DataFrame({"feature": [1, 2, 3]}),
            pd.DataFrame({"feature": [4, 5]}),
            pd.Series([1, 2, 3]),
            pd.Series([4, 5])
        )

        get_mock = Mock()
        get_mock.fit.return_value = get_mock

        mock_get_model.return_value = get_mock

        model = XgBoost()

        with self.assertWarns(UserWarning) as warn:
            model.train(self.dtf_data)
        self.assertIn("Attenzione: nessun parametro specificato per il modello.", str(warn.warning))


    @patch("modello.XgBoost._get_model")
    def test_pred_return_model(self,mock_train):

        get_mock = Mock()
        get_mock.fit.return_value = get_mock

        mock_train.return_value = get_mock

        model = XgBoost()
        model.train(self.dtf_data)
        model.predict(self.dtf_pred)
        self.assertEqual(model.model, get_mock)

    def test_pred_model_none(self):

        model = XgBoost()
        with self.assertRaises(AttributeError) as error:
            model.predict(self.dtf_pred)
        self.assertIn("Non è possibile eseguire 'predict'", str(error.exception))

    @patch("modello.XgBoost._get_model")
    def test_save_file_path_equal_self(self, mock_train):
        get_mock = Mock()
        get_mock.fit.return_value = get_mock

        mock_train.return_value = get_mock

        model = XgBoost()
        model.train(self.dtf_data)

        check = ModelForTest()

        self.assertEqual(model.file_path, check.file_path)


    def test_save_directory_not_exist(self):
        model = XgBoost()
        #allora se metto la patch prima non va, la ragione è che c'è il controllo del costruttore
        # che agisce prima di quello in save, ergo mi blocca il path dato che l'ho patchato, devo patchare
        #dopo aver chiamato il costruttore

        with patch("modello.os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                model.save()

    @patch("builtins.open", new_callable=mock_open)
    @patch("modello.os.path.exists", return_value=True)
    @patch("pickle.dump")
    def test_save_saved(self,mock_pickel, mock_path_exist, mock_open_cal):

        model = XgBoost()
        model.save(filepath="fake_path")

        mock_open_cal.assert_called_once_with("fake_path", "wb")
        mock_pickel.assert_called_once()


    @patch("modello.pickle.load")
    @patch("builtins.open", new_callable=mock_open, read_data="fake_data")
    @patch("modello.os.path.exists", return_value = True)
    def test_load_model_return_model(self,mock_exists, mock_open_func, fake_model_load):
        fake_model = ModelForTest()
        fake_model_load.return_value = fake_model

        loaded_model = XgBoost().load("fake_path")
        self.assertEqual(loaded_model, fake_model)

if __name__ == '__main__':
    unittest.main()
