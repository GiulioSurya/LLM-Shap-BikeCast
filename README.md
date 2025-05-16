
# 🚲 XGB Bike Predictor

A modular and scalable Python project for predicting bike rentals using an XGBoost regressor and a custom preprocessing pipeline.

> ⚠️ **Disclaimer**: This project is focused on **writing clean, modular, production-ready code**, rather than on business KPIs or model performance optimization.  
> Its main goal is to provide a structured, extensible pipeline for preprocessing, training, SHAP-based interpretation, and deployment via REST APIs and LLMs.

---

## 📌 Overview

This repository includes:

* A fully object-oriented and reusable preprocessing pipeline.
* Modular feature engineering components (e.g., binning, categorical encoding, custom time features).
* An extensible `Model` class with an `XgBoost` concrete implementation.
* Grid search with exportable Excel reports.
* SHAP integration for interpretability.
* LLM-based textual explanations of predictions (via [Ollama](https://ollama.com)).
* FastAPI endpoints for training, prediction, tuning, and interpretation.
* Model and pipeline serialization with `pickle`.

---

## 🧩 Module Breakdown

### 🛠 `preprocessing.py`
Implements the data preprocessing logic using modular transformation classes. Each transformation is reusable and testable.

### 🔄 `postprocessing.py`
Handles post-model transformations, like feature binning (e.g., humidity, windspeed) for SHAP interpretability.

### 📦 `modell.py`
Defines the abstract `Model` class and a concrete `XgBoost` implementation. Handles training, grid search, prediction, SHAP computation, saving/loading.

### 🧬 `schemas.py`
Defines Pydantic classes to validate data for each API endpoint (e.g., training, prediction, interpretation).

### 🌐 `API2.py`
Implements FastAPI endpoints:
- `POST /grid_search`: run grid search and export results to Excel.
- `POST /training`: train and save an XGBoost model.
- `POST /predict`: predict rental demand using a trained model.
- `POST /interpret`: generate a **natural language interpretation** of a prediction using SHAP + LLM.

> ✅ All endpoints are stable and functional.

### 🧠 `llm.py`
Wraps the logic for generating natural language interpretations:
- Selects top SHAP features.
- Loads examples from `examples.json`.
- Creates prompts enriched with SHAP values and mapped descriptions.
- Sends prompts to Ollama and parses LLM responses.

### ✅ `unittest_model.py`
Comprehensive unit tests for:
- Model initialization
- Grid search input validation
- Training/prediction logic
- File I/O for saving and loading
- Output validation and coverage of edge cases

---

## 🧠 LLM Setup (Ollama)

To enable LLM-based prediction interpretation:

### 1. Install Ollama
Follow platform-specific setup from [https://ollama.com](https://ollama.com)

### 2. Pull the Required Model
Run the following from terminal:

```bash
ollama pull llama3.1:8b
````

You may change the default model in `llm.py`, but make sure the corresponding model is downloaded.

---

## 🚀 Running the API

Install required packages (listed in `requirements.txt`) and:

```bash
pip install uvicorn
```

Start the FastAPI server:

```bash
uvicorn modello.API2:api --reload --port 8000
```

Access interactive docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Files Included

* `hour.csv`: Original dataset for experiments
* `jsons/`:

  * `mapping.json`: Descriptions and category mappings for each feature
  * `examples.json`: Few-shot examples to guide LLM responses
* `requirements.txt`: Dependency list for reproducibility

---

## 💡 Final Notes

This project is ideal for **data scientists and developers** looking to:

* Build and deploy modular ML pipelines
* Incorporate SHAP for model transparency
* Add interpretability using LLMs
* Interface everything via clean API design

Feel free to contribute or ⭐ the repo if it helped you!

```

