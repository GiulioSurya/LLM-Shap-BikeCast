import os.path
from modello.preprocessing import Preprocessing
import pandas as pd
from modello.modell import XgBoost
from modello.llm import GenerateInterpretation
from fastapi import HTTPException, FastAPI
from modello.schemas import GridSearchParams, TrainingParams, PredictionParams, InterpretationParams

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
    model_fit.save(filepath = params.save_path)

@api.post("/predict")
def predict(params: PredictionParams):

    dtf_prect = pd.read_csv(params.data_pred_path)

    preprocessor = Preprocessing()
    preprocessor.transform_data(pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\PythonProject1\datas\train.csv"), "train")

    dtf_prect_cln = preprocessor.transform_data(dtf_prect, "predict")
    dtf_prect_cln.drop(columns=[params.target], inplace=True)

    model_fit = XgBoost().load(params.load_model)
    model_pred = model_fit.predict(dtf_prect_cln)

    return model_pred.tolist()


@api.post("/interpret")
def call_interpret(params: InterpretationParams):

    #carica i dati
    df_train = pd.read_csv(params.data_train_path) #train
    df_input = pd.DataFrame([params.raw_input]) #input
    #carica modello
    model = XgBoost.load(filepath=params.model_path)

    #preprocessing dei dati
    preproc = Preprocessing()
    df_train_encod = preproc.transform_data(df_train, "train") #train
    df_input_encod = preproc.transform_data(df_input, "predict") #input
    # Allinea l'ordine delle colonne con quello del training
    df_input_encod = df_input_encod[df_train_encod.columns.drop('cnt')]

    # calcola predizione
    pred_value = model.predict(df_input_encod)[0]

    #shap values
    shap_df = model.shap_values(df_train_encod, df_input_encod)
    shap_dict = shap_df.iloc[0].to_dict()

    #genera interpretazione
    interpreter = GenerateInterpretation(
        pred_input=df_input_encod,
        pred_value=pred_value,
        shap_dict=shap_dict,
        n_variable=4
    )

    explanation = interpreter.get_interpretation()

    return {
        "predicted_value": float(pred_value),
        "interpretation": explanation
    }





















