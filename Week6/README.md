#CelebalTechAssignmentWeek6
# House Price Prediction – Model Evaluation & Hyperparameter Tuning

This project evaluates and tunes multiple regression models to predict house sale prices. Using the processed data from a previous preprocessing pipeline, this project compares model performances, applies hyperparameter tuning, and visualizes the results to select the best-performing model.

---

## 📁 Project Structure
<br>
CelebalTechAssignmentWeek6 <br>
├── data/ <br>
│ └── processed_train.csv # Cleaned dataset (features + target) <br>
├── model_train.py # Full pipeline script <br>
├── results/ <br>
│ ├── metrics_summary.csv # Metrics from base models <br>
│ ├── tuned_metrics_summary.csv # Metrics from tuned models <br>
│ └── tuning_results.csv # Best parameters for each model <br>
├── visuals/<br>
│ ├── residuals_<model>.png # Residual histogram<br>
│ ├── actual_vs_predicted_<model>.png # Scatterplot of true vs predicted<br>
│ ├── r2_score_comparison.png # Barplot: R² of all models<br>
│ ├── rmse_comparison.png # Barplot: RMSE of all models<br>
│ └── best_model_confusion.png # Final highlight of best model<br>
├── house_modeling.ipynb # Jupyter notebook version with code & output<br>
├── requirements.txt # Required libraries<br>
├── LICENSE # MIT License<br>
└── README.md # This file<br>
---


## 🚀 Workflow Overview

### Step 1: Load Data
- Load `processed_train.csv`
- Split into `X` (features) and `y` (target)
- Create train-test split (80/20)

### Step 2: Define Models
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Support Vector Regressor (SVR)  
- K-Nearest Neighbors (KNN)  
- Linear Regression

### Step 3: Model Evaluation
- Custom evaluation function to compute:
  - R² Score  
  - RMSE  
  - MAE  
- Saves visualizations:
  - Residual distribution
  - Predicted vs actual plots

### Step 4: Hyperparameter Tuning
- GridSearchCV or RandomizedSearchCV applied to all models
- Optimal parameters stored and evaluated again

### Step 5: Results Summary
- Performance metrics for both base and tuned models saved
- Comparison plots generated
- Best model automatically highlighted and visualized

---

## 📊 Key Metrics Used

| Metric         | Description                       |
|----------------|-----------------------------------|
| R² Score       | Goodness of fit                   |
| RMSE           | Root Mean Squared Error           |
| MAE            | Mean Absolute Error               |

---

## 📈 Visual Outputs

| File                                  | Description                             |
|---------------------------------------|-----------------------------------------|
| `residuals_<model>.png`              | Histogram of prediction residuals       |
| `actual_vs_predicted_<model>.png`    | Scatterplot of predictions vs ground truth |
| `r2_score_comparison.png`            | R² scores of all tuned models           |
| `rmse_comparison.png`                | RMSE of all tuned models                |
| `best_model_confusion.png`           | Highlighted visual for top model        |

---

## ⚙️ Hyperparameter Tuning Summary

Each model was tuned using either `GridSearchCV` or `RandomizedSearchCV` to optimize performance. The best parameter combinations were recorded and evaluated again to assess their impact.

### Saved Outputs

| File                                | Description                                        |
|-------------------------------------|----------------------------------------------------|
| `results/tuning_results.csv`        | Best hyperparameters for each tuned model          |
| `results/tuned_metrics_summary.csv`| Final metrics (R², RMSE, MAE) for all tuned models |

These files allow easy comparison between base and optimized models. The tuning results can be used directly in future training pipelines or deployed systems.
---

## 🔍 Model Analysis

All models were trained and evaluated using consistent data and metrics. Based on the tuned performance:

- **Top Models (by R² Score):**
  - XGBoost
  - Gradient Boosting
  - Random Forest

- **Insights:**
  - XGBoost consistently delivered the best generalization, with the highest R² score and lowest RMSE.
  - Gradient Boosting and Random Forest performed closely behind, with slightly higher error margins.
  - SVR and KNN struggled to capture complex feature interactions, resulting in underperformance.
  - Linear Regression showed high bias, indicating the need for more expressive models for this dataset.

---

## 🏆 Best Model: XGBoost

The **XGBoost Regressor** emerged as the best-performing model based on:

| Metric     | Value              |
|------------|--------------------|
| R² Score   | 0.912628458        |
| RMSE       | 25887.6075757712   |
| MAE        | 15705.0353435359   |

### Visualization:

| Visual                               | Description                              |
|--------------------------------------|------------------------------------------|
| ![visuals/best_model_confusion.png](visuals/best_model_confusion.png)   | Actual vs Predicted plot for XGBoost     |

This plot shows how closely the model's predictions align with actual house prices. A well-aligned diagonal line indicates high accuracy and minimal bias.

---


