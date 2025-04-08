# 🚲 XGB Bike Predictor

A modular and scalable Python project for predicting bike rentals using an XGBoost regressor and a custom preprocessing pipeline.  
⚠️ **Note**: This project is not focused on business case analysis or model performance interpretation, but rather on **creating clean, production-ready, and extensible Python code** for preprocessing, training, and deployment.

---

## 📌 Overview

This repository provides:

- A fully object-oriented preprocessing pipeline.
- Modular feature engineering components (e.g., wind binning, temporal encoding).
- A custom `Model` class structure, with full XGBoost support.
- Grid search for hyperparameter tuning and result export to Excel.
- Serialization of both model and pipeline using `pickle`.
- Example training and prediction scripts in a reproducible pipeline.

---
## 🧩 Project Modules Overview

This repository is organized into modular components, each serving a specific purpose in the XGBoost-based bike demand prediction system:

### 📦 `preprocessing.py`
Handles data preprocessing, including feature engineering, transformations, and preparation of the dataset for training and evaluation.

### ⚙️ `models.py`
Defines the core model logic. It includes a base `Model` class and a specific implementation using `XGBoost`. Handles training, prediction, model saving/loading, and evaluation.

### 📐 `schemas.py`
Contains Pydantic data schemas used to validate and structure input/output data when working with the API. Ensures consistent and reliable data exchange.

### 🌐 `API2.py`
Implements a FastAPI-based RESTful API that exposes endpoints for:
- Grid search of model hyperparameters
- Model training
- Making predictions on new data

### 🚀 `usage.py`
Provides example code for end-to-end execution: from preprocessing to model training and predictions. Useful for testing and demonstration purposes.

### ✅ `unittest_model.py`
Includes unit tests to validate the functionality of the model and the preprocessing pipeline. Ensures robustness and helps catch regressions.

### 📄 `requirements.txt`
Lists all Python dependencies required to run the project. Use `pip install -r requirements.txt` to install them.

### 📊 `hour.csv`
Sample dataset used to test and demonstrate the model. Contains historical bike rental data with environmental and seasonal features.

---

This modular structure makes the project easy to maintain, extend, and deploy. Feel free to explore each module and adapt it to your own predictive modeling use case!


## 🐝 API

To run the API, you need to install **uvicorn**.

Launch the API from the terminal using:

uvicorn API2:api --reload --port XXXX

Replace XXXX with any available port number (e.g., 8000, 5000, etc.).

⚠️ Important:
Make sure to adjust import paths in the source code if they don’t match your local directory structure.
The current paths reflect the setup on the original developer’s machine and may need to be modified for your environment.



