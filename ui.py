"""CSS and small HTML builders for the app."""
from __future__ import annotations

import html

import streamlit as st

esc = html.escape

CSS = """
<style>
:root{
  --ink:#0f172a; --muted:#64748b; --faint:#94a3b8; --line:#e2e8f0; --soft:#f8fafc;
  --brand:#2563eb; --brand-soft:#eff6ff; --ok:#15803d; --ok-bg:#f0fdf4; --warn:#b45309; --warn-bg:#fffbeb;
}

/* hide Streamlit chrome */
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
.stDeployButton, [data-testid="stDecoration"]{display:none !important;}
[data-testid="stHeader"]{background:transparent !important;}
div[data-testid="stElementContainer"]:has(style), div.element-container:has(style){display:none;}

/* page width */
.block-container, [data-testid="stMainBlockContainer"]{
  max-width:1240px !important; margin:0 auto !important;
  padding:2.2rem 1.5rem 3rem !important;
}
[data-testid="stVerticalBlock"]{gap:.8rem;}

/* header */
.hero{margin:0 0 1.1rem;}
.hero-title{font-size:1.5rem;font-weight:700;color:var(--ink);line-height:1.2;}
.hero-sub{font-size:.92rem;color:var(--muted);margin-top:.15rem;}

/* cards */
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px !important;background:#fff;}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px;
  box-shadow:0 1px 2px rgba(15,23,42,.04);}
.card-h{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;margin-bottom:.6rem;
  font-size:.74rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);}
.card-h-r{font-weight:500;letter-spacing:0;text-transform:none;color:var(--faint);font-size:.76rem;text-align:right;}
.sec-title{font-size:.74rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
  color:var(--muted);margin:.1rem 0 .15rem;}

/* schematic */
.schem{display:block;margin:.2rem auto;max-width:100%;max-height:400px;width:auto;height:auto;}
.legend{display:flex;flex-wrap:wrap;justify-content:center;gap:4px 14px;margin-top:.5rem;
  font-size:.78rem;color:#334155;}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;border:1px solid #334155;
  margin-right:5px;vertical-align:-1px;}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:.7rem;}
.chip{background:var(--soft);border:1px solid var(--line);border-radius:999px;padding:3px 11px;
  font-size:.78rem;color:#475569;}
.chip b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums;}

/* parameter rows */
.p-lab{line-height:1.25;}
.p-name{display:block;font-size:.88rem;font-weight:600;color:var(--ink);}
.p-rng{display:block;font-size:.72rem;color:var(--faint);font-variant-numeric:tabular-nums;}

/* sliders */
[data-testid="stSlider"]{padding:0 .35rem;}
[data-testid="stSliderTickBar"], [data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"]{display:none !important;}
[data-testid="stSliderThumbValue"]{font-size:.72rem !important;font-weight:700 !important;
  color:var(--brand) !important;font-variant-numeric:tabular-nums;}
[data-testid="stNumberInput"] input{font-variant-numeric:tabular-nums;font-weight:600;
  text-align:right;padding-right:.4rem;}
[data-testid="stNumberInput"] button{border-radius:8px;}

/* results */
.res-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;}
.res-card{min-width:0;background:#fff;border:1px solid var(--line);border-radius:14px;
  padding:14px 16px;box-shadow:0 1px 2px rgba(15,23,42,.04);}
.res-model{font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:var(--brand);margin-bottom:.55rem;}
.res-name{font-size:.8rem;color:var(--muted);}
.res-val{font-size:1.85rem;font-weight:700;color:var(--ink);line-height:1.15;
  font-variant-numeric:tabular-nums;letter-spacing:-.01em;}
.res-unit{font-size:.9rem;font-weight:500;color:var(--muted);margin-left:.3rem;letter-spacing:0;}
.res-sub{font-size:.72rem;color:var(--faint);margin:.1rem 0 .75rem;}
.res-sub:last-child{margin-bottom:0;}
.pill{display:block;border-radius:10px;padding:.5rem .75rem;font-size:.8rem;margin-top:.6rem;line-height:1.35;}
.pill.ok{background:var(--ok-bg);color:var(--ok);border:1px solid #bbf7d0;}
.pill.warn{background:var(--warn-bg);color:var(--warn);border:1px solid #fde68a;}
.pill.info{background:var(--brand-soft);color:#1d4ed8;border:1px solid #bfdbfe;}
.fine{font-size:.76rem;color:var(--faint);line-height:1.45;margin-top:.5rem;}

/* sticky preview/results column (desktop only) */
[data-testid="stColumn"]:has(.sticky-anchor), [data-testid="column"]:has(.sticky-anchor){
  position:sticky;top:3.4rem;align-self:flex-start;
  max-height:calc(100vh - 4.4rem);overflow-y:auto;overflow-x:hidden;padding-right:2px;}
@media (max-width:640px){
  [data-testid="stColumn"]:has(.sticky-anchor), [data-testid="column"]:has(.sticky-anchor){
    position:static;max-height:none;overflow:visible;}
  .block-container, [data-testid="stMainBlockContainer"]{padding:1.2rem .9rem 2.5rem !important;}
  .res-val{font-size:1.6rem;}
}

.foot{color:var(--faint);font-size:.78rem;text-align:center;margin-top:2.2rem;line-height:1.5;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero():
    st.markdown(
        '<div class="hero"><div class="hero-title">FEA Surrogate Model</div>'
        '<div class="hero-sub">Max von Mises stress and max deformation from 2D plane-stress '
        'models</div></div>', unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f'<div class="sec-title">{esc(text)}</div>', unsafe_allow_html=True)


def sticky_anchor():
    """Marks the enclosing column as sticky (see CSS)."""
    st.markdown('<span class="sticky-anchor"></span>', unsafe_allow_html=True)


def fmt_num(x: float) -> str:
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}"
    if ax >= 0.01:
        return f"{x:.4f}"
    return f"{x:.3g}"


def legend_html(items) -> str:
    """items: iterable of (label, colour)."""
    spans = "".join(f'<span><i class="sw" style="background:{c}"></i>{esc(lbl)}</span>'
                    for lbl, c in items)
    return f'<div class="legend">{spans}</div>'


def preview_card(svg_uri: str, caption: str, chips, legend: str = "") -> str:
    chips_html = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in chips)
    return (f'<div class="card"><div class="card-h"><span>Configuration preview</span>'
            f'<span class="card-h-r">{esc(caption)}</span></div>'
            f'<img class="schem" alt="Schematic of the current configuration" src="{svg_uri}"/>'
            f'{legend}<div class="chips">{chips_html}</div></div>')


def _err_txt(err):
    return f"median error ≈ {err:.0f}%" if err is not None else "&nbsp;"


def results_html(results, agreement=None) -> str:
    """results: list of dicts(model, vm_mpa, vm_err, defl_mm, defl_err).
    agreement: optional (level, text) with level in {'ok','warn'}."""
    cards = ""
    for r in results:
        cards += (
            '<div class="res-card">'
            f'<div class="res-model">{esc(r["model"])}</div>'
            '<div class="res-name">Max von Mises stress</div>'
            f'<div class="res-val">{fmt_num(r["vm_mpa"])}<span class="res-unit">MPa</span></div>'
            f'<div class="res-sub">{_err_txt(r["vm_err"])}</div>'
            '<div class="res-name">Max deformation</div>'
            f'<div class="res-val">{fmt_num(r["defl_mm"])}<span class="res-unit">mm</span></div>'
            f'<div class="res-sub">{_err_txt(r["defl_err"])}</div>'
            '</div>')
    out = f'<div class="res-grid">{cards}</div>'
    out += '<div class="fine">Median error = typical relative error on held-out test cases.</div>'
    if agreement:
        level, text = agreement
        out += f'<span class="pill {level}">{esc(text)}</span>'
    return out


def kpi_html(tiles) -> str:
    """tiles: list of (title, big_value, sub_text)."""
    cards = "".join(
        f'<div class="res-card"><div class="res-model">{esc(t)}</div>'
        f'<div class="res-val">{esc(v)}</div><div class="res-sub">{esc(s)}</div></div>'
        for t, v, s in tiles)
    return f'<div class="res-grid">{cards}</div>'


def pill(level: str, text: str) -> str:
    """Standalone note pill; level in {'ok', 'info', 'warn'}."""
    return f'<span class="pill {level}">{esc(text)}</span>'


def footer():
    st.markdown('<div class="foot">Surrogate estimates, not a live FEA solve. Inputs are limited to '
                'the training ranges.</div>', unsafe_allow_html=True)
