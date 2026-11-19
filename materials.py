"""
Material library and the E-scaling used for the rib bracket.

The rib model is trained on Structural Steel (E = 200 GPa). The part is linear elastic, loaded by
forces only, with a fixed face, so stress does not depend on E and deformation scales as 1/E:

    deformation = deformation_steel * (E_REF / E)        stress = stress_steel

Poisson's ratio has a small effect. Checked against 51 ANSYS runs on Aluminium 6061-T6
(E = 68.9 GPa, nu = 0.33), validation_assets/ansys_aluminium_check.csv:
  * deformation: mean error 0.09 %, max 0.75 % against the 200/E scaling
  * stress: unchanged within 0.05 % where steel stress > 100 MPa; low-stress designs
    (< 100 MPa) come out up to ~4 % higher in aluminium (max at the support corner).
"""
from __future__ import annotations

E_REF_GPA = 200.0      # Young's modulus of the material the surrogate was trained on
NU_REF = 0.30

# name -> (E [GPa], nu, density [kg/m3], verified in ANSYS?)
# Steel and aluminium use exactly the values entered in ANSYS Workbench; the others are
# typical handbook values and are handled through the same scaling law.
MATERIALS = {
    "Structural Steel (ANSYS default)": (200.0, 0.30, 7850.0, True),
    "Aluminium 6061-T6": (68.9, 0.33, 2700.0, True),
    "Stainless Steel 304": (193.0, 0.31, 7750.0, False),
    "Titanium Ti-6Al-4V": (113.8, 0.342, 4430.0, False),
    "Grey Cast Iron": (110.0, 0.28, 7200.0, False),
    "Copper": (117.0, 0.34, 8300.0, False),
}
DEFAULT_MATERIAL = "Structural Steel (ANSYS default)"

E_BOUNDS_GPA = (20.0, 400.0)
NU_BOUNDS = (0.15, 0.45)


def deformation_scale(E_gpa: float) -> float:
    """Factor applied to the steel-trained deformation prediction."""
    return E_REF_GPA / float(E_gpa)


def matches_preset(name: str, E_gpa: float, nu: float, tol: float = 1e-6) -> bool:
    if name not in MATERIALS:
        return False
    e, n, _, _ = MATERIALS[name]
    return abs(e - E_gpa) <= tol * max(e, 1.0) and abs(n - nu) <= 1e-9 + tol
