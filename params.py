"""
Parameter definitions and the combined slider + number-box widget.

Single source of truth for ranges is sampling.py (the bounds the surrogates were
trained on). Lengths are shown to the user in millimetres and converted back to
metres (SI) before they reach the models.

Widget sync design
------------------
Every parameter has one authoritative value in st.session_state["v::<pid>"].
The slider ("s::<pid>") and the number box ("n::<pid>") are just two views of
it: whichever one the user touches updates the authoritative value in its
on_change callback, and on the following rerun both widgets are re-seeded from
it. Because the authoritative value is NOT a widget key, it survives switching
geometry (Streamlit discards state of widgets that are not rendered).
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

from sampling import PLATE_BOUNDS, BEAM_BOUNDS, RIB_BOUNDS


@dataclass(frozen=True)
class Param:
    pid: str            # unique id, e.g. "plate.L"
    key: str            # feature / argument name, e.g. "L"
    label: str
    lo: float           # bounds in UI units
    hi: float
    default: float
    step: float
    unit: str = ""
    scale: float = 1.0  # UI value * scale = SI value
    decimals: int = 2
    hint: str = ""

    def si(self, ui_value: float) -> float:
        return ui_value * self.scale


def _make(geom, key, label, rng, step, *, mm=False, unit="", default=None,
          decimals=2, hint=""):
    mult = 1000.0 if mm else 1.0
    lo, hi = round(rng[0] * mult, 6), round(rng[1] * mult, 6)
    dflt = round((lo + hi) / 2.0, 6) if default is None else float(default)
    return Param(pid=f"{geom}.{key}", key=key, label=label, lo=lo, hi=hi,
                 default=dflt, step=step, unit="mm" if mm else unit,
                 scale=1.0 / mult, decimals=decimals, hint=hint)


# ---------------------------------------------------------------- plate
_pb = PLATE_BOUNDS
PLATE_GROUPS = [
    ("Plate dimensions", [
        _make("plate", "L", "Length  L", _pb["L"], 1.0, mm=True),
        _make("plate", "W", "Width  W", _pb["W"], 0.5, mm=True),
        _make("plate", "t", "Thickness  t", _pb["t"], 0.1, mm=True),
    ]),
    ("Hole", [
        _make("plate", "D_frac", "Diameter ratio  D/W", _pb["D_frac"], 0.01,
              hint="Hole diameter as a fraction of the plate width."),
        _make("plate", "x_off_frac", "X-offset", _pb["x_off_frac"], 0.01, default=0.0,
              hint="Fraction of the maximum clearance along the length (0 = centred)."),
        _make("plate", "y_off_frac", "Y-offset", _pb["y_off_frac"], 0.01, default=0.0,
              hint="Fraction of the maximum clearance across the width (0 = centred)."),
    ]),
    ("Loading", [
        _make("plate", "P", "Load  P", _pb["P"], 100.0, unit="N", decimals=0),
        _make("plate", "theta_deg", "Load angle  θ", _pb["theta_deg"], 0.5, unit="deg",
              default=0.0, hint="Angle from the plate axis."),
    ]),
]

# ---------------------------------------------------------------- beam
_bb = BEAM_BOUNDS
BEAM_GROUPS = [
    ("Beam dimensions", [
        _make("beam", "L", "Length  L", _bb["L"], 1.0, mm=True),
        _make("beam", "W", "Width  W", _bb["W"], 0.5, mm=True),
        _make("beam", "t", "Thickness  t", _bb["t"], 0.1, mm=True),
    ]),
    ("Holes (2)", [
        _make("beam", "D_frac", "Diameter ratio  D/W", _bb["D_frac"], 0.01,
              hint="Hole diameter as a fraction of the beam width."),
        _make("beam", "spacing_frac", "Hole spacing", _bb["spacing_frac"], 0.01, default=0.5,
              hint="Fraction of the valid spacing range (0 = closest, 1 = furthest apart)."),
        _make("beam", "x_center_frac", "Hole group position", _bb["x_center_frac"], 0.01,
              default=0.0, hint="Fraction of the valid range along the beam (−1 = root, +1 = free end)."),
        _make("beam", "y_off_frac", "Y-offset", _bb["y_off_frac"], 0.01, default=0.0,
              hint="Fraction of the maximum clearance across the width (0 = centred)."),
    ]),
    ("Loading", [
        _make("beam", "P", "Load  P", _bb["P"], 100.0, unit="N", decimals=0),
        _make("beam", "theta_deg", "Load angle  θ", _bb["theta_deg"], 0.5, unit="deg",
              default=15.0, hint="Axial + transverse load combination."),
    ]),
]

# ---------------------------------------------------------------- rib bracket
_rb = RIB_BOUNDS
RIB_GROUPS = [
    ("Wall plate & arm", [
        _make("rib", "H", "Wall plate height  H", _rb["H"], 1.0, mm=True),
        _make("rib", "L_arm", "Arm length  L", _rb["L_arm"], 1.0, mm=True),
        _make("rib", "t_wall", "Wall plate thickness", _rb["t_wall"], 0.1, mm=True),
        _make("rib", "t_arm", "Arm thickness", _rb["t_arm"], 0.1, mm=True),
        _make("rib", "b", "Bracket width  b", _rb["b"], 0.5, mm=True,
              hint="Out-of-plane width of the bracket."),
    ]),
    ("Gusset rib", [
        _make("rib", "rib_w_frac", "Rib length along arm", _rb["rib_w_frac"], 0.01,
              default=0.7, hint="Fraction of the arm length."),
        _make("rib", "rib_h_frac", "Rib height down wall", _rb["rib_h_frac"], 0.01,
              default=0.6, hint="Fraction of the free wall height below the arm."),
    ]),
    ("Loading", [
        _make("rib", "a_frac", "Load position", _rb["a_frac"], 0.01, default=0.85,
              hint="Along the arm: 0 = at the wall, 1 = at the tip."),
        _make("rib", "P", "Load  P", _rb["P"], 50.0, unit="N", decimals=0),
        _make("rib", "theta_deg", "Load angle  θ", _rb["theta_deg"], 0.5, unit="deg",
              default=10.0, hint="From straight down; positive tilts the load away from the wall."),
    ]),
]

PARAM_GROUPS = {"plate": PLATE_GROUPS, "beam": BEAM_GROUPS, "rib": RIB_GROUPS}


def all_params(geom):
    return [p for _, group in PARAM_GROUPS[geom] for p in group]


# ---------------------------------------------------------------------------
# widget plumbing
# ---------------------------------------------------------------------------
def _clamp(p: Param, v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return p.default
    return min(max(v, p.lo), p.hi)


def _commit(p: Param, widget_key: str):
    raw = st.session_state.get(widget_key)
    if raw is None:      # box was cleared mid-edit: keep the last good value
        return
    st.session_state[f"v::{p.pid}"] = _clamp(p, raw)


def _on_slider(p: Param):
    _commit(p, f"s::{p.pid}")


def _on_number(p: Param):
    _commit(p, f"n::{p.pid}")


def reset_params(geom: str):
    """on_click callback: restore every parameter of one geometry to its default."""
    for p in all_params(geom):
        st.session_state[f"v::{p.pid}"] = p.default


def param_row(p: Param) -> float:
    """Render  [label | slider | number box]  and return the value in UI units."""
    val = _clamp(p, st.session_state.get(f"v::{p.pid}", p.default))
    st.session_state[f"v::{p.pid}"] = val
    st.session_state[f"s::{p.pid}"] = val   # re-seed both views from the single source
    st.session_state[f"n::{p.pid}"] = val

    unit = f" {p.unit}" if p.unit else ""
    rng = f"{p.lo:g} – {p.hi:g}{unit}"
    tip = html.escape(p.hint or p.label, quote=True)

    c_lab, c_sld, c_num = st.columns([2.3, 3.2, 1.7], gap="small", vertical_alignment="center")
    with c_lab:
        st.markdown(
            f'<div class="p-lab" title="{tip}"><span class="p-name">{html.escape(p.label)}</span>'
            f'<span class="p-rng">{html.escape(rng)}</span></div>',
            unsafe_allow_html=True)
    with c_sld:
        st.slider(p.label, min_value=p.lo, max_value=p.hi, step=p.step,
                  format=f"%.{p.decimals}f", key=f"s::{p.pid}",
                  on_change=_on_slider, args=(p,), label_visibility="collapsed")
    with c_num:
        st.number_input(p.label, min_value=p.lo, max_value=p.hi, step=p.step,
                        format=f"%.{p.decimals}f", key=f"n::{p.pid}",
                        on_change=_on_number, args=(p,), label_visibility="collapsed",
                        help=f"Type a value between {p.lo:g} and {p.hi:g}{unit} and press Enter. "
                             "Range = what the surrogate models were trained on.")
    return val
