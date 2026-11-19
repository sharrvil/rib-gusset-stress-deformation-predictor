# FEA Surrogate Model — Streamlit app

Instant max von Mises stress & max deformation for a plate with a hole, a beam with
holes (XGBoost / MLP trained on scikit-fem 2D plane-stress FEA) and a rib / gusset
bracket (**Gaussian-process model trained only on ANSYS Mechanical results**, ANSYS Structural Steel, E = 200 GPa; other materials via an exact E-scaling law).

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Files
| File | Purpose |
|---|---|
| `app.py` | Page layout |
| `params.py` | Parameter ranges and slider / number-box widget |
| `schematics.py` | SVG schematics |
| `surrogate.py` | Model loading and prediction |
| `materials.py` | Material list and E-scaling |
| `ansys_check.py` | Rib inputs as Workbench parameter values |
| `ansys_model.py` | Wrapper class needed to load the rib `.joblib` models |
| `train_ansys.py` | Trains the rib model from the ANSYS table |
| `build_training_table.py` | Workbench export CSV to training table |
| `ui.py` | CSS and HTML helpers |
| `sampling.py` | Parameter bounds and feature engineering |
| `validation_assets/ansys_rib_results.csv` | 469 ANSYS design points (rib training data) |
| `validation_assets/ansys_aluminium_check.csv` | 51 ANSYS runs on Al 6061-T6 vs steel |
| `models/` | Trained models |

## Rib model (ANSYS only)
* Data: 469 usable ANSYS cases (DP 0/1 duplicate baselines and the failed DP 129 are dropped).
* Accuracy (5-fold cross-validation, out-of-fold): deformation mean 1.9 % / median 1.3 % / P90 4.2 %;
  von Mises mean 8.8 % / median 6.8 % / P90 19.1 %.
* Trained on Structural Steel. The app has a material dropdown with editable E and ν: deformation = steel prediction × 200 GPa / E, stress unchanged (checked against 51 ANSYS aluminium runs in `validation_assets/ansys_aluminium_check.csv`: deformation error mean 0.09 % / max 0.75 %; stress unchanged within 0.05 % above 100 MPa, up to about 4 % higher below that). See `materials.py`.
* The "Compare with ANSYS" section lists the P1–P13 values for the current inputs and compares your ANSYS result with the prediction (`ansys_check.py`).
* Full explanation of the training pipeline and how to retrain: `TRAINING_GUIDE.md`.
* Load position `a_frac` is limited to 0.55–0.90 (the ANSYS load patch must stay on the arm).

Retrain after changing the CSV: `python train_ansys.py` (needs scikit-learn, matplotlib).
Rebuilding the CSV from a Workbench export: `build_training_table.py` (see its docstring).

**Caveats**: 2D plane stress, Mechanical 2026 R1 (student), quadratic elements, face sizing =
min(t_wall, t_arm)/2.5, back face of the wall plate fixed, load on a patch of 0.15·L on the arm
top face. Max von Mises at the sharp rib corners is a stress singularity and mesh-dependent, so
stress is less accurate than deformation and indicative only.
