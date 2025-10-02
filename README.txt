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
----------------------------------------------------

🚀 How to Use:

# Clone the repository from GitHub
!git clone https://github.com/riadbabba-commits/Riad_et_al_2025.git

# Move into the project folder
%cd Riad_et_al_2025

import numpy as np
import joblib

# Load the pre-trained models
lr_model = joblib.load("LinearRegression_model_Riad_et_al_2025.pkl")
dt_model = joblib.load("DecisionTree_model_Riad_et_al_2025.pkl")
rf_model = joblib.load("RandomForest_model_Riad_et_al_2025.pkl")
xgb_model = joblib.load("XGBoost_model_Riad_et_al_2025.pkl")

# Example input (replace with your own values)
# newdata = np.array([[SiO2/Al2O3, NaOH, NS/NH, T]])
newdata = np.array([[4.6, 8, 2, 50]])

# Make predictions with all models
lr_pred = lr_model.predict(newdata)
dt_pred = dt_model.predict(newdata)
rf_pred = rf_model.predict(newdata)
xgb_pred = xgb_model.predict(newdata)

# Display results
print("Predicted Compressive Strength at 28 days (CS28, MPa):")
print("Linear Regression →", lr_pred[0])
print("Decision Tree →", dt_pred[0])
print("Random Forest →", rf_pred[0])
print("XGBoost →", xgb_pred[0])

----------------------------------------------
