# Trained Machine Learning Models (Riad et al., 2025)

This repository contains trained models and training parameters used in the study.

## 📂 Files
- `LinearRegression_model_Riad_et_al_2025.pkl`
- `DecisionTree_model_Riad_et_al_2025.pkl`
- `RandomForest_model_Riad_et_al_2025.pkl`
- `XGBoost_model_Riad_et_al_2025.pkl`
- `training_params_Riad_et_al_2025.json`

## ⚙️ Requirements
- Python 3.10+
- Libraries: `scikit-learn`, `xgboost`, `joblib`, `pandas`, `numpy`

Install them using:
```bash
pip install -r requirements.txt


🚀 How to Use:

import joblib
import pandas as pd

# Load model
rf_model = joblib.load("RandomForest_model_Riad_et_al_2025.pkl")

# Example input (replace with your data)
X_new = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=["feat1","feat2","feat3","feat4"])

# Predict
y_pred = rf_model.predict(X_new)
print("Prediction:", y_pred)
