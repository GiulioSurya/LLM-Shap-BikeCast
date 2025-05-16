import re
import ollama
from typing import Dict
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from modello.preprocessing import Preprocessing
from modello.modell import XgBoost
from modello.postprocessing import Postprocessing




class GenerateInterpretation(object):

    def __init__(self,
                 pred_input: pd.DataFrame,
                 pred_value: float,
                 shap_dict: Dict[str,float],
                 n_variable: int = 4):

        self.pred_input = Postprocessing().transform(pred_input)
        self.pred_value = pred_value
        self.shap_dict = shap_dict
        self.n_variable = n_variable

    def get_interpretation(self,
            model_name: str = "llama3.1:8b"
    ) -> str:
        """
        Chiama l'LLM per ottenere l'interpretazione testuale completa,

        a partire da un oggetto GenerateInterpretation.

        Parameters:
            interpreter: istanza di GenerateInterpretation già configurata
            model_name: nome del modello Ollama

        Returns:
            Risposta testuale generata dal modello
        """
        str_prompt = self._generate_prompt()

        response = ollama.generate(
            model=model_name,
            prompt=str_prompt,
            options={
                "temperature": 0.2,
                "num_predict": 500,
                "num_ctx": 10000,
                "top_p": 0.7
            }
        )

        str_response = response.get("response", "").strip()

        match = re.search(r"(.*?)&", str_response, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return str_response


    def _generate_prompt(self) -> str:
        """
        Genera un prompt per LLM che contiene:
        - contesto e istruzioni
        - esempi few-shot
        - valore predetto
        - descrizione e valore osservato delle 4 variabili più rilevanti (da self._sort_shap)
        """
        top_vars = [k for k, _ in self._sort_shap()]
        descriptions = self._get_description()
        mappings = self._get_mapping()
        pred_values = self.pred_input

        str_prompt = (
            "Riceverai un valore predetto e 4 variabili che lo hanno influenzato, "
            "ognuna accompagnata da una descrizione e dal valore osservato.\n"
            "Il tuo compito è generare una spiegazione testuale, coerente e fluida, nello stile degli esempi seguenti:\n\n"
        )

        # Inserisci gli esempi
        str_prompt += self._get_examples()

        # Sezione finale da interpretare
        str_prompt += "\n\n---\n\n"
        str_prompt += f"Valore predetto: {round(self.pred_value)} biciclette\n"
        str_prompt += "Variabili rilevanti:\n"

        for var in top_vars:
            valore_grezzo = pred_values[var].values[0]
            descrizione = descriptions.get(var, "N/A")
            mapping_dict = mappings.get(var, {})
            valore_mappato = mapping_dict.get(str(int(valore_grezzo)), str(valore_grezzo)) if isinstance(mapping_dict,
                                                                                                         dict) else str(
                valore_grezzo)

            str_prompt += f"- {var}:\n"
            str_prompt += f"  • Descrizione: {descrizione}\n"
            str_prompt += f"  • Valore osservato: {valore_mappato}\n"

        str_prompt += "\nGenera ora una spiegazione coerente nello stile degli esempi precedenti."
        return str_prompt

    def _sort_shap(self):
        """
        Ordina i valori SHAP per importanza assoluta decrescente.
        Restituisce una lista delle top 4 tuple (variabile, valore SHAP).
        """
        sorted_shap = sorted(self.shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_shap = sorted_shap[:self.n_variable]
        return sorted_shap

    def _get_description(self) -> Dict[str, str]:
        """
        Restituisce un dizionario {variabile: mappatura} per le top 4 variabili
        ordinate per importanza SHAP, usando il file mapping.json.
        """
        # Carica il mapping
        with open("jsons/mapping.json", "r", encoding="utf-8") as f:
            mapping = json.load(f)

        # Prendi le chiavi ordinate
        top_vars = [k for k, _ in self._sort_shap()]

        # Estrai descrizioni dal mapping
        result = {var: mapping[var]["description"] for var in top_vars if var in mapping}

        return result

    def _get_mapping(self):
        """
        Restituisce un dizionario {variabile: descrizione} per le top 4 variabili
        ordinate per importanza SHAP, usando il file mapping.json.
        """
        """
                Restituisce un dizionario {variabile: descrizione} per le top 4 variabili
                ordinate per importanza SHAP, usando il file mapping.json.
                """
        # Carica il mapping
        with open("jsons/mapping.json", "r", encoding="utf-8") as f:
            mapping = json.load(f)

        # Prendi le chiavi ordinate
        top_vars = [k for k, _ in self._sort_shap()]

        # Estrai descrizioni dal mapping
        result = {var: mapping[var]["values"] for var in top_vars if var in mapping}

        return result

    def _get_examples(self):

        with open("jsons/examples.json",encoding="utf-8") as f:
            examples = json.load(f)

        examples_str = "\n\n".join(f"{k}:\n{v}" for k, v in examples.items())

        return examples_str




if __name__ == "__main__":
    # 1) Carica dati
    df = pd.read_csv("datas/hour.csv")
    df_train, df_predict = train_test_split(df, test_size=0.2, random_state=42)

    # 2) Preprocessing
    preproc = Preprocessing()
    df_train_enc = preproc.transform_data(df_train, "train")
    df_pred_enc  = preproc.transform_data(df_predict, "predict")

    # 3) Carica modello già salvato
    model = XgBoost.load(
        filepath=r"training/mio_modello.pkl"
    )

    # 4) Predizione
    # Rimuovo la colonna target per predict
    df_pred_input = df_pred_enc.drop(columns=[model.target_col])
    preds = model.predict(df_pred_input)


    # 5) SHAP su singola istanza (qui indice 0)
    shap_df = model.shap_values(df_train_enc, df_pred_input.iloc[[1]])

    # 6) Estrai dizionario SHAP per l’istanza 0
    shap_dict = shap_df.iloc[0].to_dict()
    pred_value = preds[1]


    test = GenerateInterpretation(pred_input=df_pred_input.iloc[[0]],
                                  pred_value=pred_value,
                                  shap_dict=shap_dict,
                                  n_variable=4)

    test.get_interpretation()

    all_interpretations = {}

    for i in range(10):
        shap_df = model.shap_values(df_train_enc, df_pred_input.iloc[[i]])
        shap_dict = shap_df.iloc[0].to_dict()
        pred_value = preds[i]

        test = GenerateInterpretation(
            pred_input=df_pred_input.iloc[[i]],
            pred_value=pred_value,
            shap_dict=shap_dict,
            n_variable=4
        )

        interpretation = test.get_interpretation()
        all_interpretations[f"instance_{i}"] = {
            "prediction": float(pred_value),
            "interpretation": interpretation
        }

    # Scrivi tutte le interpretazioni in un singolo file
    with open("jsons/all_interpretations.json", "w", encoding="utf-8") as f:
        json.dump(all_interpretations, f, ensure_ascii=False, indent=4)



