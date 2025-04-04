import sys
from pathlib import Path
from modello.preprocessing import Preprocessing
import pandas as pd
from modello.modell import XgBoost
from pydantic import BaseModel, Field
from fastapi import FastAPI
from typing import Dict, List, Optional, Union

api = FastAPI()


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
        title="Nome della metrica di scoring",
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
                        "n_estimators": [400],
                        "max_depth": [7],
                        "learning_rate": [0.05],
                        "min_child_weight": [65],
                        "subsample": [0.7],
                        "colsample_bynode": [0.7],
                        "reg_lambda": [0.5]
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
        ..., title="Percorso ai dati per il training"
    )
    model_parameter: Dict[str, List[float]] = Field(
        ..., title="Parametri per il training"
    )
    target: Optional[str] = Field(
        default="cnt", title="Nome della colonna target"
    )
    kwargs: Optional[dict] = Field(
        default=None, title="Parametri aggiuntivi per il training"
    )


class PredictionParams(BaseModel):
    data_pred_path: Union[str, Path] = Field(
        ..., title="Percorso ai dati per la predizione"
    )
    target: Optional[str] = Field(
        default=None, title="Nome della colonna target"
    )




@api.post("/grid_search")
def grid_search(params: GridSearchParams):

    dtf_grid = pd.read_csv(params.data_grid_path)
    preproc = Preprocessing()
    dtf_grid_cln = preproc.transform_data(dtf_grid, "train")

    XgBoost(file_path=params.save_path).grid_search(
        dtf_data=dtf_grid_cln,
        grid_params=params.grid_parameters,
        file_name=params.file_name,
        target_col=params.target_col,
        scoring=params.scoring,
        metrics=params.metrics,
        **(params.kwargs or dict)
    )















