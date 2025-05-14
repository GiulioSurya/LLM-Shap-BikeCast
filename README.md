# 🚲 XGB Bike Predictor

A modular and scalable Python project for predicting bike rentals using an XGBoost regressor and a custom preprocessing pipeline.

> ⚠️ **Disclaimer**: This project is focused on **writing clean, modular, production-ready code**, rather than on business KPIs or model performance optimization.
> Its main goal is to provide a structured, extensible pipeline for preprocessing, training, SHAP-based interpretation, and deployment.

---

## 📌 Overview

This repository includes:

* A fully object-oriented and reusable preprocessing pipeline.
* Modular feature engineering components (e.g., binning, categorical encoding, custom time features).
* An extensible `Model` class, including XGBoost-specific functionality.
* Support for grid search with result export to Excel.
* SHAP integration for model interpretability.
* Natural language explanations of predictions via LLMs.
* Serialization of both the model and preprocessing pipeline with `pickle`.
* FastAPI endpoints for external usage (⚠️ **still under development**).

---

## 🧩 Module Breakdown

### 🛠 `preprocessing.py`

Manages preprocessing logic with a chain of transformations (e.g., scaling, encoding, new variable creation). Every transformation is implemented as a separate class for flexibility.

### 🔄 `postprocessing.py`

Performs postprocessing on prediction inputs before passing them to interpretation modules (e.g., SHAP explanation binning).

### 📦 `modell.py`

Contains the abstract `Model` class and a concrete `XgBoost` implementation. Handles training, grid search, prediction, SHAP computation, and model persistence.

### 🧬 `schemas.py`

Defines Pydantic schemas used to validate inputs and outputs for FastAPI. Ensures structured and validated data.

### 🌐 `API2.py`

Implements RESTful endpoints via FastAPI:

* `/grid_search`: to tune model hyperparameters
* `/training`: to train and persist the model
* `/predict`: to make predictions on new data

> ⚠️ **Note**: These endpoints are a work in progress. Minor refactoring or adjustments may still be needed.

### 🧠 `llm.py`

Uses an LLM (via **[Ollama](https://ollama.com/)**) to generate natural language interpretations of predictions based on SHAP values and feature descriptions.

### ✅ `unittest_model.py`

Provides unit tests for key components in the model and preprocessing logic. Helps ensure robustness, consistency, and easier debugging.

---

## 🧠 LLM Setup (Ollama)

To enable the LLM-based interpretation of predictions, the project requires **Ollama** with a specific model.

### 1. Install Ollama

Download and install from [https://ollama.com](https://ollama.com) — follow the instructions for your operating system.

### 2. Pull the Required Model

Once installed, run the following command to download the required model:

```bash
ollama pull llama3.1:8b
```

The model used in the pipeline defaults to `llama3`. If you modify the `model_name` parameter in `llm.py`, make sure to pull the corresponding model.

---

## 🚀 Running the API

Install the required dependency:

```bash
pip install uvicorn
```

Launch the FastAPI server:

```bash
uvicorn API2:api --reload --port 8000
```

Then visit: [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger documentation.

> 🔧 **Note**: You may need to adjust import paths in the source code depending on your local file structure. Paths reflect the original developer’s environment.

---

## 📁 Files Included

* `hour.csv`: Sample dataset for training and testing.
* `requirements.txt`: List of dependencies.
* `jsons/`: Folder expected to contain:

  * `examples.json`: Few-shot examples for prompt generation
  * `mapping.json`: Descriptions and categorical mappings for each variable

---

## 💡 Final Notes

This project is designed for **developers and data scientists** looking for a robust and modular ML pipeline with integrated model explainability.

Feel free to fork, extend, or customize according to your needs — and ⭐ the repo if you find it useful!


