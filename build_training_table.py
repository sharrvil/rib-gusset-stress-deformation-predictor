"""
Turn a Workbench "Export Design Points" CSV into the training table for the app.

    python build_training_table.py  <workbench_export.csv>  [material_key.csv | -]  [out.csv]

* Parameter IDs below (P1..P17) are the ones in your project; if Workbench numbered
  anything differently, edit the PID block.
* Drops DP 0 / DP 1 (identical baseline copies), any row whose outputs are blank
  (failed design points) and exact duplicate inputs.
* Material: the table is for ONE material (ANSYS Structural Steel, E = 200 GPa). The key file is optional;
  rows missing from it are labelled "Structural Steel". Other materials are NOT mixed into the training
  table: they follow the E-scaling law in materials.py (verified with ANSYS runs on aluminium).
"""
import sys
import numpy as np
import pandas as pd

PID = dict(L_plus_tw="P1", tw="P2", rib_back="P3", tip_to_patch="P5", ta="P6", rib_top="P7",
           H="P8", patch_w="P9", b="P10", mesh="P11", Fx="P12", Fy="P13",
           defl="P14", vm="P15", Rx="P16", Ry="P17")


def build(export_csv, key_csv):
    d = pd.read_csv(export_csv, comment="#")
    key = (pd.read_csv(key_csv).set_index("Name")["Material"].to_dict()
           if key_csv and key_csv != "-" else {})
    n0 = len(d)
    d = d[~d.Name.isin(["DP 0", "DP 1"])]
    d = d.dropna(subset=[PID["defl"], PID["vm"]])
    p = {k: d[v] for k, v in PID.items()}
    H, tw, ta = p["H"], p["tw"], p["ta"]
    L = p["L_plus_tw"] - tw
    out = pd.DataFrame({
        "dp": d.Name, "H_mm": H, "L_arm_mm": L, "t_wall_mm": tw, "t_arm_mm": ta, "b_mm": p["b"],
        "rib_w_mm": L - p["rib_back"], "rib_h_mm": H - p["rib_top"] - ta,
        "a_frac": (L - (p["tip_to_patch"] + p["patch_w"] / 2)) / L,
        "P_N": np.hypot(p["Fx"], p["Fy"]), "theta_deg": np.degrees(np.arctan2(p["Fx"], -p["Fy"]))})
    mat = d.Name.map(key).fillna("Structural Steel")
    out["material_wall"] = out["material_arm"] = out["material_rib"] = mat
    out["ansys_max_deformation_mm"] = p["defl"]
    out["ansys_max_von_mises_MPa"] = p["vm"]
    out["ansys_mesh_size_mm"] = p["mesh"]
    geo = ["H_mm", "L_arm_mm", "t_wall_mm", "t_arm_mm", "b_mm", "rib_w_mm", "rib_h_mm", "a_frac", "P_N", "theta_deg"]
    out = out.drop_duplicates(subset=geo + ["material_wall"]).reset_index(drop=True)
    print(f"{n0} rows read -> {len(out)} usable;", out.material_wall.value_counts().to_dict())
    return out


if __name__ == "__main__":
    exp = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else "-"
    res = build(exp, key)
    res.to_csv(sys.argv[3] if len(sys.argv) > 3 else "ansys_rib_results.csv", index=False)
