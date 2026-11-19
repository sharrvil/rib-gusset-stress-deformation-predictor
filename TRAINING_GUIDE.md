# How the rib / gusset model is trained (and how to retrain it yourself)

## The idea
Instead of solving the bracket in ANSYS every time, we solve it ~470 times with different dimensions and
loads, then teach a regression model the pattern "inputs -> max stress, max deformation". The trained model
answers instantly. Only the rib bracket uses ANSYS data; the plate and beam use their own (scikit-fem) models.

## Pipeline
1. **Choose the design space** (`sampling.py`, `RIB_BOUNDS`): ranges for wall height H, arm length, wall and arm
   thickness, width b, rib size (as fractions so every sample is a valid shape), load position, load P, load angle.
2. **Sample it** with a Latin hypercube (`scipy.stats.qmc.LatinHypercube`): ~470 well-spread points instead of a
   grid, which would need thousands of runs.
3. **Run each point in ANSYS Workbench.** The Parameter Set holds the inputs P1-P13 (H10, H12, ... forces) and the
   outputs P14 (Total Deformation max) and P15 (Equivalent Stress max). *Update All Design Points*, then
   export the table (right-click the Table of Design Points -> export table data) to CSV.
4. **Build the training table** (`build_training_table.py`): converts Workbench columns into physical inputs
   (L_arm = H10 - H12, rib_w = L_arm - H14, rib_h = V9 - V13 - V11, load P and angle from Fx/Fy, ...),
   drops the duplicate baselines (DP 0 / DP 1) and rows with blank outputs (failed solves).
   Result: `validation_assets/ansys_rib_results.csv` (469 usable rows).
5. **Feature engineering** (`rib_engineer_features`): dimensions and load -> logarithms, rib size -> ratios,
   angle -> sin/cos. Stress and deflection behave like products of powers of the inputs, which are almost
   linear in log space, so this makes the learning problem much easier.
6. **Model** (`train_ansys.py`): a **Gaussian process** (Matern 5/2 kernel, one length-scale per input = ARD)
   plus a noise term, on standardised features and **log-transformed targets**. It is a good fit for a few
   hundred smooth, expensive-to-generate samples. One model for deformation, one for stress.
7. **Honest accuracy**: 5-fold cross-validation. Each case is predicted by a model that never saw it
   ("out-of-fold"). Reported: median error ~1.3 % (deformation), ~6.8 % (stress). The final model is then refit on
   all cases and saved to `models/model_rib_<target>_ansys.joblib`.
8. **The app** loads those files; `predict()` builds the same features from the sliders.

## Material
The model is trained on ONE material (Structural Steel, E = 200 GPa, nu = 0.3). For a linear-elastic part loaded
by forces, stress does not depend on E and deformation scales as 1/E, so other materials use
`deformation = steel prediction * 200 / E`, `stress = steel prediction` (`materials.py`). Checked with 51 ANSYS
aluminium runs (Al 6061-T6, E = 68.9 GPa, nu = 0.33): `validation_assets/ansys_aluminium_check.csv`.
Do not mix materials into the training CSV; that would only add noise.

## Retrain it yourself
Requirements: `pip install -r requirements.txt -r requirements_train.txt`.

1. Put your Workbench export in a folder, e.g. `my_export.csv` (ANSYS Structural Steel only).
2. `python build_training_table.py my_export.csv - validation_assets/ansys_rib_results.csv`
   (the `-` means "no material key file"). Check the printed "rows read -> usable" line.
3. `python train_ansys.py` - trains, prints the accuracy, rewrites the two `.joblib` models, the validation CSV,
   the parity plot and the rib rows of `model_performance_summary.csv`.
4. `streamlit run app.py` - the app picks up the new models automatically.

### Adding more data
Run more design points in Workbench (new LHS samples inside the ranges), export, append the rows, and repeat
steps 2-3. Most useful: more points where the error is large  and near the edges of the ranges.

### Changing the ranges
Edit `RIB_BOUNDS` in `sampling.py`; the sliders update automatically. Re-run ANSYS for the new region, because the
model should not be trusted outside what it was trained on.

### Things that will break it
* A different parameter numbering in Workbench: edit the `PID` block in `build_training_table.py`.
* A different geometry or boundary condition: the rib model is only valid for this bracket and load patch
  (0.15 x arm length), fixed back face, plane stress.
* Versions: `scikit-learn==1.8.0` is pinned because the `.joblib` files are tied to it; retraining on a different
  version is fine as long as the app uses the same one.
