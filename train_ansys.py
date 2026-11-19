"""
Train the rib / gusset bracket surrogate on ANSYS Mechanical results only.

    python train_ansys.py

Reads   validation_assets/ansys_rib_results.csv   (ANSYS design points, ANSYS Structural Steel, E = 200 GPa)
Writes  models/model_rib_<target>_ansys.joblib     Gaussian process (Matern 5/2, ARD, log targets)
        validation_assets/model_performance_summary.csv   (rib rows replaced)
        validation_assets/validation_samples_rib.csv      every case with its out-of-fold prediction
        validation_assets/pred_vs_actual_rib.png          parity plot
Accuracy is measured out-of-fold (5-fold cross-validation): no case is predicted by a
model that saw it during training. The saved model is then refit on all cases.
Needs scikit-learn, matplotlib (training only).
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ansys_model import ANSYS_FEATURES, LogTargetModel
from sampling import rib_engineer_features

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
VAL, MODELS = os.path.join(HERE, "validation_assets"), os.path.join(HERE, "models")

TARGETS = {  # target -> (ANSYS column, to-SI factor, unit)
    "max_deformation": ("ansys_max_deformation_mm", 1e-3, "mm"),
    "max_von_mises": ("ansys_max_von_mises_MPa", 1e6, "MPa")}


def make_gp():
    n = len(ANSYS_FEATURES)
    kernel = (ConstantKernel(1.0) * Matern(np.full(n, 3.0), (1e-2, 1e3), nu=2.5)
              + WhiteKernel(0.05, (1e-6, 1.0)))
    return make_pipeline(StandardScaler(),
                         GaussianProcessRegressor(kernel, normalize_y=True,
                                                  n_restarts_optimizer=5, random_state=0))


def raw_inputs(df):
    """ANSYS table (mm) -> SI raw inputs for the shared feature function."""
    return pd.DataFrame({
        "H": df.H_mm / 1e3, "L_arm": df.L_arm_mm / 1e3, "t_wall": df.t_wall_mm / 1e3,
        "t_arm": df.t_arm_mm / 1e3, "b": df.b_mm / 1e3, "rib_w": df.rib_w_mm / 1e3,
        "rib_h": df.rib_h_mm / 1e3, "a_frac": df.a_frac, "P": df.P_N, "theta_deg": df.theta_deg})


def main():
    df = pd.read_csv(os.path.join(VAL, "ansys_rib_results.csv"))
    raw = raw_inputs(df)
    X = rib_engineer_features(raw)[ANSYS_FEATURES].values
    n = len(df)
    summary, oof = [], {}
    for target, (col, to_si, unit) in TARGETS.items():
        y = df[col].values
        yl = np.log(y)
        pred = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=1).split(X):
            pred[te] = np.exp(make_gp().fit(X[tr], yl[tr]).predict(X[te]))
        oof[target] = pred
        rel = np.abs(pred - y) / y * 100
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2l = 1 - ((yl - np.log(pred)) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()
        summary.append(dict(geometry="rib", target=target, model="ansys", R2=r2, R2_log=r2l,
                            **{"mean_rel_err_%": rel.mean(), "median_rel_err_%": np.median(rel),
                               "p90_rel_err_%": np.percentile(rel, 90)},
                            n_train=int(n * 0.8), n_test=n))
        final = make_gp().fit(X, yl)
        joblib.dump({"model": LogTargetModel(final, to_si), "features": ANSYS_FEATURES},
                    os.path.join(MODELS, f"model_rib_{target}_ansys.joblib"))
        print(target, "mean %.1f%% median %.1f%% p90 %.1f%%  R2 %.3f" %
              (rel.mean(), np.median(rel), np.percentile(rel, 90), r2))

    # summary file: keep plate / beam rows, replace rib rows
    sp = os.path.join(VAL, "model_performance_summary.csv")
    old = pd.read_csv(sp)
    pd.concat([old[old.geometry != "rib"], pd.DataFrame(summary)]).to_csv(sp, index=False)

    # every case with its out-of-fold prediction, SI units like the plate / beam sample files
    s = raw.copy()
    s["max_von_mises"] = df.ansys_max_von_mises_MPa * 1e6
    s["max_deformation"] = df.ansys_max_deformation_mm * 1e-3
    for target, (col, to_si, unit) in TARGETS.items():
        s[f"pred_max_{target[4:]}_ansys"] = oof[target] * to_si
        s[f"rel_err_max_{target[4:]}_ansys_%"] = np.abs(oof[target] - df[col].values) / df[col].values * 100
    s.insert(0, "dp", df.dp.values)
    s.to_csv(os.path.join(VAL, "validation_samples_rib.csv"), index=False)
    parity_plot(df, oof)


def parity_plot(df, oof):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=150)
    for ax, (target, (col, _, unit)) in zip(axes, TARGETS.items()):
        x, p = df[col].values, oof[target]
        lo, hi = min(x.min(), p.min()) * 0.8, max(x.max(), p.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect prediction")
        ax.scatter(x, p, s=10, alpha=.5, lw=0, color="#1f77b4")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), aspect="equal",
               xlabel=f"ANSYS actual ({unit})", ylabel=f"Predicted ({unit})",
               title=f"{target} — ANSYS-trained GP (5-fold CV)")
        ax.legend(fontsize=8)
    fig.suptitle("Predicted vs Actual — rib / gusset bracket (ANSYS)")
    fig.tight_layout()
    fig.savefig(os.path.join(VAL, "pred_vs_actual_rib.png"), facecolor="white")


if __name__ == "__main__":
    main()
