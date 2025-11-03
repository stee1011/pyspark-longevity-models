# 🧬 RobustLife-ML

An end-to-end machine learning pipeline exploring **life expectancy prediction** using **PySpark**, **Pandas**, and **Scikit-Learn**, with an experimental **adversarial robustness test** that simulates Gaussian noise to evaluate model stability.

---

## 🚀 Overview

This project demonstrates:

* **PySpark-based ETL** for high-volume data ingestion.
* **Feature cleaning**, imputation, and outlier isolation using `IsolationForest`.
* **Hybrid ML pipelines** with categorical + numerical preprocessing via `ColumnTransformer`.
* **Model benchmarking** across multiple regressors.
* **Adversarial distortion analysis** to test model overfitting & resilience.

---

## 🧠 Models Compared

| Model                           | Description            | Strength                          |
| ------------------------------- | ---------------------- | --------------------------------- |
| **Random Forest Regressor**     | Ensemble bagging trees | High accuracy, robust under noise |
| **Gradient Boosting Regressor** | Sequential boosting    | Excellent bias-variance tradeoff  |
| **Linear Regression**           | Baseline linear fit    | Simple, interpretable             |
| **Huber Regressor**             | Robust linear model    | Resistant to outliers             |

---

## 📊 Results Snapshot

### ✅ Before Noise

| Model            | MAE  | MSE   | R²       |
| ---------------- | ---- | ----- | -------- |
| RandomForest     | 1.09 | 3.37  | **0.95** |
| GradientBoosting | 1.11 | 3.36  | **0.95** |
| LinearRegression | 2.53 | 10.35 | 0.84     |
| HuberRegressor   | 2.51 | 10.43 | 0.84     |

### ⚠️ After Noise Injection

| Model            | MAE  | MSE   | R²   |
| ---------------- | ---- | ----- | ---- |
| RandomForest     | 3.75 | 23.72 | 0.64 |
| GradientBoosting | 3.83 | 24.29 | 0.63 |
| LinearRegression | 5.30 | 36.46 | 0.45 |
| HuberRegressor   | 5.65 | 40.51 | 0.39 |

🧩 Ensemble models show **superior robustness** under Gaussian feature corruption.

---

## 🧰 Stack

* **PySpark** – distributed preprocessing
* **Pandas / Seaborn** – visualization and data wrangling
* **Scikit-Learn** – pipelines, transformers, ensemble regressors
* **Matplotlib** – exploratory data plots

---

## 🧪 Experiment Insight

> Injected Gaussian + Chi²-like noise into test features to mimic real-world data drift and evaluate resilience.

**Result:**
RandomForest and GradientBoosting maintained predictive integrity far better than linear models, validating their use for life expectancy estimation in non-stationary data environments.

---

## ⚙️ Setup

```bash
# Clone repository
git clone https://github.com/stee1011/robustlife-ml.git
cd robustlife-ml

# Create environment
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt
```

---

## 🧾 License

MIT License © 2025

---

## 👨‍💻 Author

**Stephen Njoroge**
*Data Science & Cybersecurity Enthusiast*

> Experiments at the intersection of **AI robustness**, **data integrity**, and **health metrics.**
