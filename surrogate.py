"""
Model loading and prediction.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

import ansys_model  # noqa: F401  (needed to unpickle the ANSYS-trained models)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
VAL_DIR = os.path.join(APP_DIR, "validation_assets")

GEOMETRIES = {"Rib / Gusset Bracket": "rib", "Plate with Hole": "plate", "Beam with Holes": "beam"}
GEOM_LABELS = {v: k for k, v in GEOMETRIES.items()}
MODEL_LABELS = {"xgboost": "XGBoost", "mlp": "MLP (Neural Net)", "ansys": "ANSYS-trained (GP)"}
TARGETS = {"max_von_mises": "Max von Mises stress", "max_deformation": "Max deformation"}



class NativeXGBModel:
    """Wraps an xgboost Booster (loaded from its JSON file) with the same predict(X) as the sklearn models."""

    def __init__(self, booster, features):
        self.booster = booster
        self.features = features

    def predict(self, X):
        dmat = xgb.DMatrix(np.asarray(X), feature_names=self.features)
        log_pred = self.booster.predict(dmat)
        return np.exp(log_pred)  # models are trained on log-transformed targets


@st.cache_resource(show_spinner="Loading surrogate models…")
def load_models():
    models = {}
    for geom in ("plate", "beam"):          # scikit-fem-trained XGBoost + MLP
        for target in TARGETS:
            booster = xgb.Booster()
            booster.load_model(os.path.join(MODEL_DIR, f"model_{geom}_{target}_xgboost.json"))
            with open(os.path.join(MODEL_DIR, f"model_{geom}_{target}_xgboost_features.json")) as f:
                feats = json.load(f)
            models[(geom, target, "xgboost")] = {"model": NativeXGBModel(booster, feats),
                                                 "features": feats}
            models[(geom, target, "mlp")] = joblib.load(
                os.path.join(MODEL_DIR, f"model_{geom}_{target}_mlp.joblib"))
    # rib bracket: trained on ANSYS results only (see train_ansys.py)
    for target in TARGETS:
        models[("rib", target, "ansys")] = joblib.load(
            os.path.join(MODEL_DIR, f"model_rib_{target}_ansys.joblib"))
    return models


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(VAL_DIR, "model_performance_summary.csv"))


def predict(self, X):
        dmat = xgb.DMatrix(np.asarray(X), feature_names=self.features)
        log_pred = self.booster.predict(dmat)
        return np.exp(log_pred)  # models are trained on log-transformed targets


@st.cache_resource(show_spinner="Loading surrogate models…")
def load_models():
    models = {}
    for geom in ("plate", "beam"):          # scikit-fem-trained XGBoost + MLP
        for target in TARGETS:
            booster = xgb.Booster()
            booster.load_model(os.path.join(MODEL_DIR, f"model_{geom}_{target}_xgboost.json"))
            with open(os.path.join(MODEL_DIR, f"model_{geom}_{target}_xgboost_features.json")) as f:
                feats = json.load(f)
            models[(geom, target, "xgboost")] = {"model": NativeXGBModel(booster, feats),
                                                 "features": feats}
            models[(geom, target, "mlp")] = joblib.load(
                os.path.join(MODEL_DIR, f"model_{geom}_{target}_mlp.joblib"))
    # rib bracket: trained on ANSYS results only (see train_ansys.py)
    for target in TARGETS:
        models[("rib", target, "ansys")] = joblib.load(
            os.path.join(MODEL_DIR, f"model_rib_{target}_ansys.joblib"))
    return models


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(VAL_DIR, "model_performance_summary.csv"))


@st.cache_data
def load_validation_samples(geom: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(VAL_DIR, f"validation_samples_{geom}.csv"))


def predict(models, geom, target, model_name, feature_dict) -> float:
    """One prediction in SI units (Pa or m)."""
    bundle = models[(geom, target, model_name)]
    x = np.array([[float(feature_dict[f]) for f in bundle["features"]]])
    return float(bundle["model"].predict(x)[0])


def median_error(summary: pd.DataFrame, geom, target, model_name):
    """Median relative error (%) on the held-out test set, or None if unavailable."""
    row = summary[(summary["geometry"] == geom) & (summary["target"] == target)
                  & (summary["model"] == model_name)]
    return float(row["median_rel_err_%"].iloc[0]) if len(row) else None
