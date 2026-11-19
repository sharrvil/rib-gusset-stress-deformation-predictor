"""Rib inputs -> Workbench parameter values (P1 = H10, ...), same IDs as build_training_table.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

PATCH_FRAC = 0.15   # load patch width = 0.15 * arm length (as in the training runs)


def workbench_inputs(raw: dict, E_gpa: float, nu: float, patch_frac: float = PATCH_FRAC) -> pd.DataFrame:
    """raw: SI dict with H, L_arm, t_wall, t_arm, b, rib_w, rib_h, a_frac, P, theta_deg."""
    H, L, tw, ta, b = raw["H"], raw["L_arm"], raw["t_wall"], raw["t_arm"], raw["b"]
    rib_w, rib_h, a, P = raw["rib_w"], raw["rib_h"], raw["a_frac"], raw["P"]
    th = np.deg2rad(raw["theta_deg"])
    patch = patch_frac * L
    mm = 1e3
    rows = [
        ("P1", "H10", (L + tw) * mm, "mm", "Sketch: total horizontal length, back face of the wall to the arm tip"),
        ("P2", "H12", tw * mm, "mm", "Sketch: wall plate thickness (horizontal)"),
        ("P3", "H14", (L - rib_w) * mm, "mm", "Sketch: horizontal gap from the rib's upper end to the arm tip"),
        ("P5", "H16", ((1 - a) * L - patch / 2) * mm, "mm", "Sketch: arm tip to the nearer edge of the load patch (top edge)"),
        ("P6", "V11", ta * mm, "mm", "Sketch: arm thickness (vertical)"),
        ("P7", "V13", (H - ta - rib_h) * mm, "mm", "Sketch: height of the rib's lower end on the wall, from the bottom of the wall"),
        ("P8", "V9", H * mm, "mm", "Sketch: wall plate height"),
        ("P9", "H15", patch * mm, "mm", "Sketch: width of the load patch on the arm's top edge"),
        ("P10", "Surface Body Thickness", b * mm, "mm", "Out-of-plane width of the bracket"),
        ("P11", "Face Sizing Element Size", min(tw, ta) / 2.5 * mm, "mm", "Mesh: Face Sizing on the surface"),
        ("P12", "Force X Component", P * np.sin(th), "N", "Force on the load patch (+X = away from the wall)"),
        ("P13", "Force Y Component", -P * np.cos(th), "N", "Force on the load patch (-Y = downwards)"),
    ]
    df = pd.DataFrame(rows, columns=["ID", "Workbench name", "Value", "Unit", "Where it goes"])
    mat = pd.DataFrame([
        ("Young's Modulus", E_gpa * 1e3, "MPa", "Engineering Data > material > Isotropic Elasticity"),
        ("Poisson's Ratio", nu, "-", "Engineering Data > material > Isotropic Elasticity")],
        columns=["Workbench name", "Value", "Unit", "Where it goes"])
    mat.insert(0, "ID", ["-", "-"])
    return pd.concat([df, mat], ignore_index=True)


def design_point_csv(table: pd.DataFrame) -> str:
    """One design-point row in the same column layout as a Workbench 'Export Table' file."""
    t = table[table["ID"].str.startswith("P")]
    return pd.DataFrame([["DP app"] + [round(float(v), 6) for v in t["Value"]]],
                        columns=["Name"] + list(t["ID"])).to_csv(index=False)
