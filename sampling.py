"""
Latin Hypercube sampling for the plate and beam parameter spaces.

Hole positions are sampled as FRACTIONS of the available geometric
clearance (not raw offsets), so every sample is guaranteed to produce a
valid, non-overlapping hole layout regardless of the other sampled
dimensions (L, W, D, spacing, ...). This avoids ever needing to reject or
retry a sample.
"""
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, scale

EPS = 1e-3  # minimum clearance (m) between a hole edge and any boundary/other hole


# ---------------------------------------------------------------------
# Plate geometry (Geometry 1): bounds are on "raw" LHS dimensions, where
# D and hole offsets are expressed as FRACTIONS that get converted to
# physical values guaranteed to respect clearances.
# ---------------------------------------------------------------------
PLATE_BOUNDS = {
    # name:        (low, high)
    "L":            (0.15, 0.30),      # m
    "W":            (0.03, 0.08),      # m
    "t":            (0.003, 0.008),    # m
    "D_frac":       (0.10, 0.45),      # D = D_frac * W
    "x_off_frac":   (-0.75, 0.75),     # fraction of max allowed x offset
    "y_off_frac":   (-0.75, 0.75),     # fraction of max allowed y offset
    "P":            (2000.0, 20000.0), # N
    "theta_deg":    (-30.0, 30.0),     # deg
}

BEAM_BOUNDS = {
    "L":            (0.30, 0.50),      # m
    "W":            (0.03, 0.06),      # m
    "t":            (0.003, 0.008),    # m
    "D_frac":       (0.10, 0.35),      # D = D_frac * W
    "spacing_frac": (0.0, 1.0),        # interpolates between min/max valid spacing
    "x_center_frac": (-1.0, 1.0),      # interpolates between min/max valid center
    "y_off_frac":   (-0.75, 0.75),     # fraction of max allowed y offset
    "P":            (1000.0, 12000.0), # N
    "theta_deg":    (-40.0, 40.0),     # deg (axial + transverse combo)
}


# ---------------------------------------------------------------------
# Rib / gusset bracket (Geometry 3): wall plate + arm + triangular rib, single
# material (ANSYS Structural Steel, E = 200 GPa). Surrogate trained on ANSYS Mechanical results.
# Other materials: deformation x (200 GPa / E), stress unchanged (see materials.py).
# Rib leg lengths are FRACTIONS of the space available, so every sample is a
# valid geometry. theta is measured from straight DOWN (-y); positive tilts
# the load away from the wall.
# ---------------------------------------------------------------------
RIB_BOUNDS = {
    "H":            (0.10, 0.18),      # m   wall plate height
    "L_arm":        (0.08, 0.16),      # m   arm length (beyond the wall plate)
    "t_wall":       (0.006, 0.012),    # m   wall plate thickness (in-plane)
    "t_arm":        (0.005, 0.010),    # m   arm thickness (in-plane)
    "b":            (0.010, 0.030),    # m   out-of-plane width of bracket
    "rib_w_frac":   (0.40, 0.95),      # rib_w = frac * L_arm
    "rib_h_frac":   (0.30, 0.90),      # rib_h = frac * (H - t_arm)
    "a_frac":       (0.55, 0.90),      # load patch centre, fraction of arm length (ANSYS patch must stay on the arm)
    "P":            (500.0, 5000.0),   # N
    "theta_deg":    (-20.0, 30.0),     # deg from vertical-down
}


def _lhs_raw(bounds: dict, n_samples: int, seed: int) -> pd.DataFrame:
    names = list(bounds.keys())
    lows = np.array([bounds[k][0] for k in names])
    highs = np.array([bounds[k][1] for k in names])
    sampler = LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n=n_samples)
    scaled = scale(unit, lows, highs)
    return pd.DataFrame(scaled, columns=names)


def sample_plate_params(n_samples: int, seed: int = 0) -> pd.DataFrame:
    """Generate n_samples valid physical parameter dicts for the plate geometry."""
    raw = _lhs_raw(PLATE_BOUNDS, n_samples, seed)

    D = raw["D_frac"] * raw["W"]
    x_off_max = (raw["L"] / 2.0 - D / 2.0 - EPS).clip(lower=0)
    y_off_max = (raw["W"] / 2.0 - D / 2.0 - EPS).clip(lower=0)
    x_off = raw["x_off_frac"] * x_off_max
    y_off = raw["y_off_frac"] * y_off_max

    out = pd.DataFrame({
        "L": raw["L"], "W": raw["W"], "t": raw["t"], "D": D,
        "x_off": x_off, "y_off": y_off,
        "P": raw["P"], "theta_deg": raw["theta_deg"],
    })
    return out


def sample_beam_params(n_samples: int, seed: int = 1, n_holes: int = 2) -> pd.DataFrame:
    """Generate n_samples valid physical parameter dicts for the beam geometry
    (n_holes fixed, default 2, for a consistent ML feature vector)."""
    raw = _lhs_raw(BEAM_BOUNDS, n_samples, seed)

    D = raw["D_frac"] * raw["W"]

    # valid spacing range for n_holes evenly spaced holes to fit on [0, L]
    # with clearance EPS at each end and between holes (see hole_centers_from_params:
    # holes span from x_center - (n-1)*spacing/2 - D/2 to x_center + (n-1)*spacing/2 + D/2)
    half_span_factor = (n_holes - 1) / 2.0
    spacing_min = D * 1.25  # safety factor above touching (spacing must be > D)
    # L - D - 2*EPS is the max total span available for (n-1)*spacing
    spacing_max = ((raw["L"] - D - 2 * EPS) / max(half_span_factor * 2, 1e-9)).clip(lower=spacing_min * 1.01)
    spacing = spacing_min + raw["spacing_frac"] * (spacing_max - spacing_min)

    half_span = half_span_factor * spacing
    x_center_min = half_span + D / 2.0 + EPS
    x_center_max = raw["L"] - half_span - D / 2.0 - EPS
    x_center = x_center_min + (raw["x_center_frac"] + 1.0) / 2.0 * (x_center_max - x_center_min)

    y_off_max = (raw["W"] / 2.0 - D / 2.0 - EPS).clip(lower=0)
    y_off = raw["y_off_frac"] * y_off_max

    out = pd.DataFrame({
        "L": raw["L"], "W": raw["W"], "t": raw["t"], "D": D,
        "n_holes": n_holes, "spacing": spacing, "x_center": x_center, "y_off": y_off,
        "P": raw["P"], "theta_deg": raw["theta_deg"],
    })
    return out


# ML feature order for the rib model (single material, so no Young's-modulus terms).
# Raw physical inputs become log / ratio / sin-cos features: stress and deflection scale
# roughly as products of powers, which are near-linear in log space.
RIB_FEATURES = ["lnH", "lnL", "lnTw", "lnTa", "lnB", "rw", "rh", "a_frac", "lnP",
                "sin_th", "cos_th"]


def rib_engineer_features(d):
    """d: dict-like or DataFrame with raw rib inputs (H, L_arm, t_wall, t_arm, b, rib_w,
    rib_h, a_frac, P, theta_deg; lengths in m). Returns a same-type object holding RIB_FEATURES."""
    th = np.deg2rad(d["theta_deg"])
    out = {
        "lnH": np.log(d["H"]), "lnL": np.log(d["L_arm"]),
        "lnTw": np.log(d["t_wall"]), "lnTa": np.log(d["t_arm"]), "lnB": np.log(d["b"]),
        "rw": d["rib_w"] / d["L_arm"], "rh": d["rib_h"] / (d["H"] - d["t_arm"]),
        "a_frac": d["a_frac"], "lnP": np.log(d["P"]),
        "sin_th": np.sin(th), "cos_th": np.cos(th),
    }
    return pd.DataFrame(out) if isinstance(d, pd.DataFrame) else out


def rib_frac_to_physical(H, L_arm, t_arm, rib_w_frac, rib_h_frac):
    """Scalar version of the rib fraction->physical conversion (for the Streamlit app)."""
    return rib_w_frac * L_arm, rib_h_frac * (H - t_arm)


def plate_frac_to_physical(L, W, D_frac, x_off_frac, y_off_frac, eps=EPS):
    """Scalar version of the plate fraction->physical conversion (for use in
    the Streamlit app, where inputs come from individual sliders)."""
    D = D_frac * W
    x_off_max = max(L / 2.0 - D / 2.0 - eps, 0.0)
    y_off_max = max(W / 2.0 - D / 2.0 - eps, 0.0)
    x_off = x_off_frac * x_off_max
    y_off = y_off_frac * y_off_max
    return D, x_off, y_off


def beam_frac_to_physical(L, W, D_frac, spacing_frac, x_center_frac, y_off_frac,
                           n_holes=2, eps=EPS):
    """Scalar version of the beam fraction->physical conversion (for use in
    the Streamlit app)."""
    D = D_frac * W
    half_span_factor = (n_holes - 1) / 2.0
    spacing_min = D * 1.25
    spacing_max = max((L - D - 2 * eps) / max(half_span_factor * 2, 1e-9), spacing_min * 1.01)
    spacing = spacing_min + spacing_frac * (spacing_max - spacing_min)
    half_span = half_span_factor * spacing
    x_center_min = half_span + D / 2.0 + eps
    x_center_max = L - half_span - D / 2.0 - eps
    x_center = x_center_min + (x_center_frac + 1.0) / 2.0 * (x_center_max - x_center_min)
    y_off_max = max(W / 2.0 - D / 2.0 - eps, 0.0)
    y_off = y_off_frac * y_off_max
    return D, spacing, x_center, y_off


if __name__ == "__main__":
    p = sample_plate_params(10, seed=0)
    print("plate samples:\n", p)
    b = sample_beam_params(10, seed=1)
    print("\nbeam samples:\n", b)

    # quick validity check: every row should build a mesh without raising
    from mesh_plate import build_plate_with_hole_mesh
    from mesh_beam import build_beam_with_holes_mesh, hole_centers_from_params

    for _, row in p.iterrows():
        build_plate_with_hole_mesh(row.L, row.W, row.D, row.x_off, row.y_off)
    print("\nall plate samples produced valid geometry")

    for _, row in b.iterrows():
        centers = hole_centers_from_params(row.L, row.W, row.D, int(row.n_holes),
                                            row.spacing, row.x_center, row.y_off)
        build_beam_with_holes_mesh(row.L, row.W, row.D, centers)
    print("all beam samples produced valid geometry")
