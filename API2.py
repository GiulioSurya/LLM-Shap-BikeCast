import os.path
from modello.preprocessing import Preprocessing
import pandas as pd
from modello.modell import XgBoost
from fastapi import HTTPException, FastAPI
from modello.schemas import GridSearchParams, TrainingParams, PredictionParams

api = FastAPI()

@api.post("/grid_search")
def grid_search(params: GridSearchParams):

    if not os.path.exists(params.data_grid_path):
        raise HTTPException(status_code=404, detail=f"Il percorso {params.data_grid_path} non esiste.")

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
        **(params.kwargs or dict())
    )

@api.post("/training")
def training(params: TrainingParams):

    dtf_train = pd.read_csv(params.data_train_path)

    preprocessor = Preprocessing()
    dtf_train_cln = preprocessor.transform_data(dtf_train, "train")

    model_fit = XgBoost(model_parameters=params.model_parameter,
                        target_col=params.target,
                        **(params.kwargs or dict())
                        )
    model_fit.train(dtf_train_cln)
    model_fit.save(params.save_model)

@api.post("/predict")
def predict(params: PredictionParams):

    dtf_prect = pd.read_csv(params.data_pred_path)

    preprocessor = Preprocessing()
    preprocessor.transform_data(pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\modello\train.csv"), "train")

    dtf_prect_cln = preprocessor.transform_data(dtf_prect, "predict")
    dtf_prect_cln.drop(columns=[params.target], inplace=True)

    model_fit = XgBoost().load(params.load_model)
    model_pred = model_fit.predict(dtf_prect_cln)

    return model_pred.tolist()



















