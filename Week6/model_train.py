# model_train.py

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# SECTION 1: Setup & Load Data
os.makedirs("results", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

data_path = "data/processed_train.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["saleprice"])
y = df["saleprice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# SECTION 2: Define Models
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "LinearRegression": LinearRegression()
}

# SECTION 3: Evaluation Function
results = {}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {
        "R2 Score": r2,
        "RMSE": rmse,
        "MAE": mae
    }

    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=30, color='steelblue')
    plt.title(f"Residual Distribution - {name}")
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.savefig(f"visuals/residuals_{name}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.title(f"Actual vs Predicted - {name}")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.tight_layout()
    plt.savefig(f"visuals/actual_vs_predicted_{name}.png", dpi=300)
    plt.close()

    print(f"{name} evaluation complete.")

# SECTION 4: Train Untuned Models
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    evaluate_model(name, model, X_train, X_test, y_train, y_test)

metrics_df = pd.DataFrame(results).T
metrics_df = metrics_df.sort_values(by="R2 Score", ascending=False)
metrics_df.to_csv("results/metrics_summary.csv")

# SECTION 5: Hyperparameter Tuning
search_spaces = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        },
        "search": "grid"
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        "search": "grid"
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, verbosity=0),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        "search": "random"
    },
    "SVR": {
        "model": SVR(),
        "params": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"]
        },
        "search": "grid"
    },
    "KNN": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 7, 9]
        },
        "search": "grid"
    },
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {},
        "search": "grid"
    }
}

tuned_results = {}
tuned_params = {}

for name, config in search_spaces.items():
    print(f"Tuning {name}...")

    if config["search"] == "grid":
        search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=0
        )
    else:
        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=10,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
            random_state=42
        )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_

    tuned_params[name] = best_params
    print(f"Best params for {name}: {best_params}")

    evaluate_model(f"{name}_tuned", best_model, X_train, X_test, y_train, y_test)
    tuned_results[name] = results[f"{name}_tuned"]

pd.DataFrame(tuned_results).T.to_csv("results/tuned_metrics_summary.csv")
pd.DataFrame(tuned_params).T.to_csv("results/tuning_results.csv")

# SECTION 6: Visualize Summary + Best Model

# R2 Score Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=tuned_results.keys(), y=[v["R2 Score"] for v in tuned_results.values()], palette="viridis", hue=None)
plt.title("R2 Score Comparison - Tuned Models")
plt.ylabel("R2 Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visuals/r2_score_comparison.png", dpi=300)
plt.close()

# RMSE Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=tuned_results.keys(), y=[v["RMSE"] for v in tuned_results.values()], palette="coolwarm", hue=None)
plt.title("RMSE Comparison - Tuned Models")
plt.ylabel("RMSE")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visuals/rmse_comparison.png", dpi=300)
plt.close()

# Highlight Best Model
best_model_name = max(tuned_results, key=lambda k: tuned_results[k]["R2 Score"])
import shutil
shutil.copyfile(
    f"visuals/actual_vs_predicted_{best_model_name}_tuned.png",
    "visuals/best_model_confusion.png"
)
print(f"Best model is: {best_model_name}")