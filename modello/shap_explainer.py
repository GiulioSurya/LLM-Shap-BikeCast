from modell import XgBoost
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing
import shap



if __name__ == "__main__":
    # Esempio di utilizzo

    df_test = pd.read_csv("datas/hour.csv")

    df_train, df_predict = train_test_split(df_test, test_size=0.2, random_state=42)

    # ------------------ pre-processing dati

    preproc = Preprocessing()
    df_train_encod = preproc.transform_data(df_train, "train")
    df_test_encod = preproc.transform_data(df_predict, "predict")

    df_test = df_test_encod.copy()
    df_test_encod.drop(columns=["cnt"], inplace=True)

    # -------------------- stima griglia
    param_grid = {
        'n_estimators': [400],
        'max_depth': [7],
        'learning_rate': [0.05],
        "min_child_weight": [65],
        "subsample": [0.7],
        "colsample_bynode": [0.7],
        "reg_lambda": [0.5],
    }

    columns = [
        "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time",
        "param_colsample_bynode", "param_learning_rate", "param_max_depth",
        "param_min_child_weight", "param_n_estimators", "param_reg_lambda",
        "param_subsample", "params", "split0_test_score", "split1_test_score",
        "split2_test_score", "mean_test_score", "std_test_score", "rank_test_score",
        "split0_train_score", "split1_train_score", "split2_train_score",
        "mean_train_score", "std_train_score"
    ]

    XgBoost().grid_search(dtf_data=df_train_encod,
                          validation_size=0.5,
                          early_stopping_rounds=5,
                          target_col="cnt",
                          grid_params=param_grid,
                          file_name="risultati_grid.xlsx",
                          scoring="neg_mean_squared_error",
                          metrics=columns
                          )

    # Esempio di train
    params = {
        'n_estimators': 400,
        'max_depth': 7,
        'learning_rate': 0.05,
        "min_child_weight": 75,
        "subsample": 0.75,
        "colsample_bynode": 0.7,
        "reg_lambda": 0.5
    }



    model = XgBoost(model_parameters=params)
    model.train(df_train_encod)
    predicted = model.predict(df_test_encod)

    explainer = shap.TreeExplainer(model.model, df_train_encod.drop(columns=["cnt"], inplace=True))
    shap_values = explainer.shap_values(df_test_encod.iloc[[0]])
