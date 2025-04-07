from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union

class GridSearchParams(BaseModel):
    data_grid_path: Optional[Union[str, Path]] = Field(
        default=r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello\train.csv",
        title="Percorso ai dati per il Grid Search",
    )
    grid_parameters: Dict[str, List[Union[int, float]]] = Field(
        ...,
        title="Parametri per il Grid Search",
    )
    file_name: str = Field(
        ...,
        title="Nome del file per il Grid Search",
    )
    scoring: str = Field(
        ...,
        title="loss function per la valutazione della griglia",
    )
    target_col: Optional[str] = Field(
        default="cnt",
        title="Nome della colonna target",
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        title="Metriche da calcolare durante il Grid Search",
    )
    save_path: Optional[Union[str, Path]] = Field(
        default=r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\training",
        title="Percorso per il salvataggio del file",
    )
    kwargs: Optional[dict] = Field(
        default=dict,
        title="Parametri aggiuntivi per il Grid Search",
    )


    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data_grid_path": r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello\train.csv",
                    "grid_parameters": {
                        "n_estimators": [],
                        "max_depth": [],
                        "learning_rate": [],
                        "min_child_weight": [],
                        "subsample": [],
                        "colsample_bynode": [],
                        "reg_lambda": []
                    },
                    "file_name": "grid_search_results.xlsx",
                    "scoring": "neg_root_mean_squared_error",
                    "target_col": "cnt",
                    "metrics": [
                        "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time",
                        "param_colsample_bynode", "param_learning_rate", "param_max_depth",
                        "param_min_child_weight", "param_n_estimators", "param_reg_lambda",
                        "param_subsample", "params", "split0_test_score", "split1_test_score",
                        "split2_test_score", "mean_test_score", "std_test_score", "rank_test_score",
                        "split0_train_score", "split1_train_score", "split2_train_score",
                        "mean_train_score", "std_train_score"
                    ],
                    "save_path": r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\training",
                    "kwargs": {
                        "cv": 3,
                        "n_jobs": -1,
                        "verbose": 1,
                        "early_stopping_rounds": 10
                    }
                }
            ]
        }
    }


class TrainingParams(BaseModel):
    data_train_path: Union[str, Path] = Field(
        r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello\train.csv",
        title="Percorso ai dati per il training"
    )
    model_parameter: Dict[str, Union[float,int]] = Field(
        ...,
        title="Parametri per il training"
    )
    target: Optional[str] = Field(
        default="cnt",
        title="Nome della colonna target"
    )
    save_model: Optional[Union[str, Path]] = Field(
        default=r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\training\train",
        title="Percorso per il salvataggio del modello"
    )
    kwargs: Optional[dict] = Field(
        default=None,
        title="Parametri aggiuntivi per il training nativi di XGBRegressor"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data_train_path": r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello\train.csv",
                    "model_parameter": {
                        'n_estimators': 400,
                        'max_depth': 7,
                        'learning_rate':  0.05,
                        "min_child_weight": 75,
                        "subsample": 0.75,
                        "colsample_bynode": 0.7,
                        "reg_lambda": 0.5
                    },
                    "target": "cnt",
                    "save_model": r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\training\train",
                    "kwargs": {
                    }
                }
            ]
        }
    }



class PredictionParams(BaseModel):
    data_pred_path: Union[str, Path] = Field(
        ...,
        title="Percorso ai dati per la predizione"
    )
    target: Optional[str] = Field(
        default=None,
        title="Nome della colonna target"
    )
    load_model: Union[str, Path] = Field(
        default=r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\training\train",
        title="Percorso per il caricamento del modello"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data_pred_path": r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello\predict.csv",
                    "target": "cnt",
                    "load_model": r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\training\train"
                }
            ]
        }
    }
