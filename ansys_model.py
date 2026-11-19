"""
Surrogate trained on ANSYS Mechanical results (rib / gusset bracket, ANSYS Structural Steel, E = 200 GPa).

Kept in its own module so the joblib-saved model can be unpickled by the app and the
training script alike.  No UI code here.
"""
from __future__ import annotations

import numpy as np

# ML features: sampling.RIB_FEATURES (single material, no modulus terms).
ANSYS_FEATURES = ["lnH", "lnL", "lnTw", "lnTa", "lnB", "rw", "rh", "a_frac", "lnP",
                  "sin_th", "cos_th"]


class LogTargetModel:
    """Wraps a fitted sklearn pipeline trained on log(target in ANSYS units).
    predict() returns SI units (m or Pa) like every other model in the app."""

    def __init__(self, pipeline, to_si: float):
        self.pipeline = pipeline
        self.to_si = to_si          # mm -> m (1e-3) or MPa -> Pa (1e6)

    def predict(self, X):
        return np.exp(self.pipeline.predict(np.asarray(X, dtype=float))) * self.to_si
