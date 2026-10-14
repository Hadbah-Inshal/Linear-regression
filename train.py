"""Trains a Linear Regression model on each dataset in config.py.
Run once:  python train.py
"""
import json
import os

import joblib
import matplotlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATASETS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

for key, cfg in DATASETS.items():
    print(f"\n=== {cfg['title']} ===")
    df = pd.read_csv(cfg["csv"]).dropna().drop_duplicates()
    target = cfg["target"]
    X, y = df.drop(columns=[target]), df[target]

    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    metrics = {
        "R2": round(r2_score(y_te, pred), 4),
        "MAE": round(mean_absolute_error(y_te, pred), 4),
        "RMSE": round(mean_squared_error(y_te, pred) ** 0.5, 4),
        "train_rows": len(X_tr),
        "test_rows": len(X_te),
    }
    print(metrics)

    features = []
    for c in X.columns:
        if c in num_cols:
            features.append({"name": c, "type": "num",
                             "min": float(X[c].min()), "max": float(X[c].max()),
                             "default": round(float(X[c].mean()), 2)})
        else:
            features.append({"name": c, "type": "cat",
                             "options": sorted(X[c].astype(str).unique().tolist())})

    joblib.dump(model, f"models/{key}.joblib")
    with open(f"models/{key}.json", "w") as f:
        json.dump({"title": cfg["title"], "target": target,
                   "features": features, "metrics": metrics}, f, indent=2)

    # Actual vs predicted plot
    plt.figure(figsize=(5, 4))
    plt.scatter(y_te, pred, alpha=0.5)
    lo, hi = min(y_te.min(), pred.min()), max(y_te.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{cfg['title']} (R2={metrics['R2']})")
    plt.tight_layout()
    plt.savefig(f"static/{key}.png", dpi=110)
    plt.close()

print("\nDone. Models saved in /models, plots in /static")
