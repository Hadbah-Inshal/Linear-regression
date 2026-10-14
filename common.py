import json

import joblib
import pandas as pd

from config import DATASETS


def load_meta(key):
    with open(f"models/{key}.json") as f:
        return json.load(f)


def predict(key, values: dict) -> float:
    """values: {feature_name: value}"""
    model = joblib.load(f"models/{key}.joblib")
    meta = load_meta(key)
    row = {}
    for feat in meta["features"]:
        v = values[feat["name"]]
        row[feat["name"]] = float(v) if feat["type"] == "num" else str(v)
    return float(model.predict(pd.DataFrame([row]))[0])


__all__ = ["DATASETS", "load_meta", "predict"]
