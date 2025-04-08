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

## 🐝 API

To run the API, you need to install **uvicorn**.

Launch the API from the terminal using:

uvicorn API2:api --reload --port XXXX

Replace XXXX with any available port number (e.g., 8000, 5000, etc.).

⚠️ Important:
Make sure to adjust import paths in the source code if they don’t match your local directory structure.
The current paths reflect the setup on the original developer’s machine and may need to be modified for your environment.



