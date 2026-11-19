"""
FEA surrogate: max von Mises stress and max deformation for a rib / gusset bracket,
a plate with a hole and a beam with holes.

    streamlit run app.py
"""
import itertools
import logging

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FEA Surrogate Model",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from sampling import (  # noqa: E402  (must come after set_page_config)
    plate_frac_to_physical, beam_frac_to_physical,
    rib_frac_to_physical, rib_engineer_features,
)
import ui  # noqa: E402
from params import PARAM_GROUPS, all_params, param_row, reset_params  # noqa: E402
from schematics import MAT_COLORS, beam_schematic, plate_schematic, rib_schematic  # noqa: E402
from surrogate import (  # noqa: E402
    GEOMETRIES, MODEL_LABELS, load_models, load_summary, median_error, predict,
)
from materials import (  # noqa: E402
    DEFAULT_MATERIAL, E_BOUNDS_GPA, E_REF_GPA, MATERIALS, NU_BOUNDS,
    deformation_scale, matches_preset,
)
import ansys_check  # noqa: E402
import fea_rib  # noqa: E402

log = logging.getLogger("fea_surrogate")
ui.inject_css()
ui.hero()

try:
    MODELS = load_models()
    SUMMARY = load_summary()
except Exception:  # pragma: no cover - deployment problem, show a clean message
    log.exception("Could not load models")
    st.error("The surrogate models could not be loaded. Please check that the `models/` and "
             "`validation_assets/` folders are present and that the installed `xgboost` / "
             "`scikit-learn` versions match `requirements.txt`.")
    st.stop()


DEFAULT_GEOMETRY = "Rib / Gusset Bracket"   # geometry shown when the app first opens


def pick(label: str, key: str, label_visibility="visible") -> str:
    """Segmented control that can never end up with 'nothing selected'."""
    options = list(GEOMETRIES.keys())
    default = DEFAULT_GEOMETRY
    sel = st.segmented_control(label, options, default=default, key=key,
                               label_visibility=label_visibility)
    sel = sel or st.session_state.get(f"{key}_last", default)
    st.session_state[f"{key}_last"] = sel
    return sel


# ----------------------------------------------------------------------
# Geometry-specific: SI inputs -> model features + preview
# ----------------------------------------------------------------------
def build_plate(v):
    D, x_off, y_off = plate_frac_to_physical(v["L"], v["W"], v["D_frac"],
                                             v["x_off_frac"], v["y_off_frac"])
    feats = dict(L=v["L"], W=v["W"], t=v["t"], D=D, x_off=x_off, y_off=y_off,
                 P=v["P"], theta_deg=v["theta_deg"])
    return dict(
        features=feats, export=dict(feats),
        svg=plate_schematic(v["L"], v["W"], D, x_off, y_off, v["P"], v["theta_deg"]),
        caption="Fixed left edge · load at right edge", legend="",
        chips=[("Hole Ø", f"{D * 1e3:.2f} mm"),
               ("Hole centre offset", f"({x_off * 1e3:+.2f}, {y_off * 1e3:+.2f}) mm")])


def build_beam(v):
    D, spacing, x_center, y_off = beam_frac_to_physical(
        v["L"], v["W"], v["D_frac"], v["spacing_frac"], v["x_center_frac"], v["y_off_frac"],
        n_holes=2)
    feats = dict(L=v["L"], W=v["W"], t=v["t"], D=D, spacing=spacing, x_center=x_center,
                 y_off=y_off, P=v["P"], theta_deg=v["theta_deg"])
    return dict(
        features=feats, export=dict(feats),
        svg=beam_schematic(v["L"], v["W"], D, spacing, x_center, y_off, v["P"], v["theta_deg"]),
        caption="Fixed root (left) · load at free end", legend="",
        chips=[("Hole Ø", f"{D * 1e3:.2f} mm"), ("Spacing", f"{spacing * 1e3:.2f} mm"),
               ("Group centre x", f"{x_center * 1e3:.1f} mm"), ("Y-offset", f"{y_off * 1e3:+.2f} mm")])


CUSTOM = "Custom material"


def _mat_on_select():
    name = st.session_state["w::mat_name"]
    st.session_state["mat::name"] = name
    if name in MATERIALS:
        st.session_state["mat::E"], st.session_state["mat::nu"] = MATERIALS[name][:2]


def _mat_on_number(key, wkey, lo, hi):
    raw = st.session_state.get(wkey)
    if raw is None:
        return
    st.session_state[key] = min(max(float(raw), lo), hi)
    name = st.session_state["mat::name"]
    if name != CUSTOM and not matches_preset(name, st.session_state["mat::E"],
                                             st.session_state["mat::nu"]):
        st.session_state["mat::name"] = CUSTOM


def reset_all(geom):
    reset_params(geom)
    if geom == "rib":
        st.session_state["mat::name"] = DEFAULT_MATERIAL
        st.session_state["mat::E"], st.session_state["mat::nu"] = MATERIALS[DEFAULT_MATERIAL][:2]


def material_controls() -> dict:
    """Material dropdown + editable E and nu. Returns {name, E_gpa, nu}.
    Authoritative values live in 'mat::*' (survive switching geometry); widgets are re-seeded."""
    ss = st.session_state
    ss.setdefault("mat::name", DEFAULT_MATERIAL)
    ss.setdefault("mat::E", MATERIALS[DEFAULT_MATERIAL][0])
    ss.setdefault("mat::nu", MATERIALS[DEFAULT_MATERIAL][1])
    options = list(MATERIALS) + [CUSTOM]
    ss["w::mat_name"] = ss["mat::name"] if ss["mat::name"] in options else CUSTOM
    ss["w::mat_E"], ss["w::mat_nu"] = float(ss["mat::E"]), float(ss["mat::nu"])
    with st.container(border=True):
        ui.section_title("Material")
        st.selectbox("Material", options, key="w::mat_name", on_change=_mat_on_select,
                     label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Young's modulus  E  (GPa)", *E_BOUNDS_GPA, step=1.0, format="%.1f",
                            key="w::mat_E", on_change=_mat_on_number,
                            args=("mat::E", "w::mat_E", *E_BOUNDS_GPA))
        with c2:
            st.number_input("Poisson's ratio  ν", *NU_BOUNDS, step=0.01, format="%.3f",
                            key="w::mat_nu", on_change=_mat_on_number,
                            args=("mat::nu", "w::mat_nu", *NU_BOUNDS))
        name = ss["mat::name"]
        st.markdown(ui.pill("info", (
            f"Trained on steel (E = {E_REF_GPA:g} GPa). Other materials: deformation × "
            f"{E_REF_GPA:g}/E = ×{deformation_scale(ss['mat::E']):.3f}, stress unchanged.")),
            unsafe_allow_html=True)
    return dict(name=name, E_gpa=float(ss["mat::E"]), nu=float(ss["mat::nu"]))


def build_rib(v, mat):
    rib_w, rib_h = rib_frac_to_physical(v["H"], v["L_arm"], v["t_arm"],
                                        v["rib_w_frac"], v["rib_h_frac"])
    raw = dict(H=v["H"], L_arm=v["L_arm"], t_wall=v["t_wall"], t_arm=v["t_arm"], b=v["b"],
               rib_w=rib_w, rib_h=rib_h, a_frac=v["a_frac"], P=v["P"], theta_deg=v["theta_deg"])
    nm = mat["name"]
    legend = ui.legend_html([(f"Bracket · {nm}", MAT_COLORS[nm])])
    return dict(
        features=rib_engineer_features(raw), raw=raw, mat=mat,
        export={**raw, "material": nm, "E_GPa": mat["E_gpa"], "nu": mat["nu"]},
        svg=rib_schematic(v["H"], v["L_arm"], v["t_wall"], v["t_arm"], rib_w, rib_h,
                          v["a_frac"], v["P"], v["theta_deg"], nm, nm, nm),
        caption="Wall back-face fixed · load on arm top", legend=legend,
        chips=[("Rib legs", f"{rib_w * 1e3:.1f} × {rib_h * 1e3:.1f} mm"),
               ("Material", f"E {mat['E_gpa']:g} GPa · ν {mat['nu']:g}")])


def rel_diff(a, b):
    return abs(a - b) / ((a + b) / 2.0) * 100.0


# ======================================================================
# PAGE
# ======================================================================
geometry_label = pick("Geometry", "geom_sel")
geom = GEOMETRIES[geometry_label]
model_keys = ["ansys"] if geom == "rib" else ["xgboost", "mlp"]

left, right = st.columns([1.2, 1], gap="large")

# ---------------- inputs ----------------
with left:
    h1, h2 = st.columns([3, 1.3], vertical_alignment="center")
    with h1:
        ui.section_title("Parameters")
    with h2:
        st.button("Reset", icon=":material/restart_alt:", type="tertiary",
                  on_click=reset_all, args=(geom,), key=f"reset_{geom}",
                  help="Restore default values")

    mat = material_controls() if geom == "rib" else None

    values = {}  # key -> SI value
    for title, group in PARAM_GROUPS[geom]:
        with st.container(border=True):
            ui.section_title(title)
            for p in group:
                values[p.key] = p.si(param_row(p))

cfg = (build_plate(values) if geom == "plate"
       else build_beam(values) if geom == "beam" else build_rib(values, mat))

# ---------------- preview + results ----------------
with right:
    ui.sticky_anchor()
    st.markdown(ui.preview_card(cfg["svg"], cfg["caption"], cfg["chips"], cfg["legend"]),
                unsafe_allow_html=True)

    ui.section_title("Predicted response")
    results = []
    try:
        rows = []
        for mkey in model_keys:
            vm = predict(MODELS, geom, "max_von_mises", mkey, cfg["features"])
            defl = predict(MODELS, geom, "max_deformation", mkey, cfg["features"])
            if geom == "rib":   # trained on steel -> exact E scaling for other materials
                defl *= deformation_scale(mat["E_gpa"])
            results.append(dict(
                model=MODEL_LABELS[mkey], vm_mpa=vm / 1e6, defl_mm=defl * 1e3,
                vm_err=median_error(SUMMARY, geom, "max_von_mises", mkey),
                defl_err=median_error(SUMMARY, geom, "max_deformation", mkey)))
            rows.append({"geometry": geom, "model": MODEL_LABELS[mkey], **cfg["export"],
                         "max_von_mises_MPa": vm / 1e6, "max_deformation_mm": defl * 1e3})

        agreement = None
        if len(results) >= 2:
            pairs = list(itertools.combinations(results, 2))
            d_vm = max(rel_diff(a["vm_mpa"], b["vm_mpa"]) for a, b in pairs)
            d_df = max(rel_diff(a["defl_mm"], b["defl_mm"]) for a, b in pairs)
            worst = max(d_vm, d_df)
            level = "ok" if worst <= 30 else "info" if worst <= 60 else "warn"
            upto = "up to " if len(results) > 2 else ""
            text = (f"Model agreement: stress differs by {upto}{d_vm:.0f}%, "
                    f"deformation by {upto}{d_df:.0f}%.")
            if level == "warn":
                text += " The models disagree strongly here — treat this configuration with caution."
            agreement = (level, text)

        st.markdown(ui.results_html(results, agreement), unsafe_allow_html=True)
        st.download_button("Download result (CSV)", data=pd.DataFrame(rows).to_csv(index=False),
                           file_name=f"fea_prediction_{geom}.csv", mime="text/csv",
                           icon=":material/download:")
    except Exception:
        log.exception("Prediction failed")
        st.error("Prediction failed for this configuration. Try resetting the parameters; "
                 "if the problem persists, check the server logs.")

    if geom == "rib":
        n_rib = int(SUMMARY[SUMMARY["geometry"] == "rib"]["n_test"].max())
        st.markdown(f'<div class="fine">Trained on {n_rib} ANSYS cases. Error shown is the '
                    'median from 5-fold cross-validation.</div>', unsafe_allow_html=True)

# ---------------- contours (rib only) ----------------
@st.cache_data(show_spinner="Solving…", max_entries=64)
def contour_figure(raw_items, E_gpa, nu, kind, target):
    return fea_rib.contour_png(dict(raw_items), E_gpa, nu, kind, target)


if geom == "rib" and results:
    st.divider()
    ui.section_title("Contours")
    raw_items = tuple(cfg["raw"].items())
    dcol, scol = st.columns(2, gap="large")
    for col, kind, title, val, unit in (
            (dcol, "deformation", "Max deformation", results[0]["defl_mm"], "mm"),
            (scol, "stress", "Max equivalent stress", results[0]["vm_mpa"], "MPa")):
        png, fmax, scale = contour_figure(raw_items, mat["E_gpa"], mat["nu"], kind, round(val, 6))
        with col:
            st.markdown(ui.kpi_html([(title, f"{fmax:.4g} {unit}", "")]), unsafe_allow_html=True)
            st.image(png, width="stretch")
    st.caption(f"Drawn on the deformed shape (×{scale:g}); dashed line is the undeformed outline. "
               "Shapes come from a 2D FEA solve, scaled to the predicted maxima.")

# ---------------- compare with ANSYS (rib only) ----------------
if geom == "rib" and results:
    st.divider()
    ui.section_title("Compare with ANSYS")
    wb = ansys_check.workbench_inputs(cfg["raw"], mat["E_gpa"], mat["nu"])
    vcol, scol = st.columns([1.5, 1], gap="large")
    with vcol:
        st.dataframe(wb, hide_index=True, width="stretch", column_config={
            "ID": st.column_config.TextColumn(width="small"),
            "Workbench name": st.column_config.TextColumn(width="medium"),
            "Value": st.column_config.NumberColumn(format="%.3f", width="small"),
            "Unit": st.column_config.TextColumn(width="small"),
            "Where it goes": st.column_config.TextColumn(width="large")})
        st.download_button("Download as design-point row (CSV)",
                           data=ansys_check.design_point_csv(wb),
                           file_name="rib_design_point.csv", mime="text/csv",
                           icon=":material/download:")
    with scol:
        res0 = results[0]
        st.markdown(ui.kpi_html([
            ("Total deformation, max", f"{res0['defl_mm']:.4g} mm", ""),
            ("Equivalent stress, max", f"{res0['vm_mpa']:.4g} MPa", "")]),
            unsafe_allow_html=True)
        a_def = st.number_input("ANSYS max deformation (mm)", value=None, min_value=0.0,
                                format="%.6g", key="ans_def")
        a_vm = st.number_input("ANSYS max stress (MPa)", value=None, min_value=0.0,
                               format="%.6g", key="ans_vm")
        bits = []
        if a_def:
            bits.append(f"Deformation: {(res0['defl_mm'] - a_def) / a_def * 100:+.1f}% vs ANSYS")
        if a_vm:
            bits.append(f"Stress: {(res0['vm_mpa'] - a_vm) / a_vm * 100:+.1f}% vs ANSYS")
        if bits:
            st.markdown(ui.pill("info", " · ".join(bits)), unsafe_allow_html=True)

ui.footer()
