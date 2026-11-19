"""
Engineering schematics for the three geometries, rendered as self-contained SVG.

* Uses matplotlib's object API (Figure) directly instead of pyplot, so no global
  figure registry builds up in a long-running server process.
* Output is a data-URI SVG (text converted to paths -> no font dependency), which
  stays razor sharp at any browser zoom level. The figure has a bounded natural
  size and the CSS (see ui.py) only ever shrinks it (max-width: 100%), so it never
  balloons when the browser is zoomed out.
* Results are memoised: identical inputs never redraw.

All lengths passed in are metres (SI); labels are printed in millimetres.
"""
from __future__ import annotations

import base64
import io
from functools import lru_cache

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["hatch.linewidth"] = 0.9

INK = "#0f172a"
EDGE = "#334155"
BODY = "#e6edf6"
HOLE = "#e11d48"
LOAD = "#2563eb"
DIM = "#64748b"
WALL = "#64748b"

from collections import defaultdict
MAT_COLORS = defaultdict(lambda: "#a9b4c2", {"Aluminium 6061-T6": "#cfd9e6", "Copper": "#d9a47a",
                                            "Titanium Ti-6Al-4V": "#b7bcc9", "Grey Cast Iron": "#8d97a5"})

FIG_W = 4.4   # max figure width  (inches)
FIG_MAX_H = 3.9  # max figure height (inches) -> tall drawings get narrower, text stays legible


# ----------------------------------------------------------------- helpers
def _canvas(xmin, xmax, ymin, ymax):
    w, h = xmax - xmin, ymax - ymin
    fig_w = min(FIG_W, FIG_MAX_H * w / h)
    fig = Figure(figsize=(fig_w, max(fig_w * h / w, 1.3)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def _encode(fig, fmt):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", pad_inches=0.06,
                transparent=True, dpi=200)
    data = buf.getvalue()
    if fmt == "svg":
        return "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")
    return data


def _hdim(ax, x0, x1, y, text, ext0=None, ext1=None, fs=11):
    """Horizontal dimension line with the label underneath."""
    for x, e in ((x0, ext0), (x1, ext1)):
        if e is not None:
            ax.plot([x, x], [e, y], color=DIM, lw=0.7, zorder=1)
    ax.annotate("", xy=(x0, y), xytext=(x1, y), zorder=2,
                arrowprops=dict(arrowstyle="<|-|>", color=DIM, lw=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    ax.annotate(text, xy=((x0 + x1) / 2, y), xytext=(0, -3), textcoords="offset points",
                ha="center", va="top", fontsize=fs, color=DIM)


def _vdim(ax, x, y0, y1, text, ext0=None, ext1=None, fs=11):
    """Vertical dimension line with a rotated label on its left."""
    for y, e in ((y0, ext0), (y1, ext1)):
        if e is not None:
            ax.plot([e, x], [y, y], color=DIM, lw=0.7, zorder=1)
    ax.annotate("", xy=(x, y0), xytext=(x, y1), zorder=2,
                arrowprops=dict(arrowstyle="<|-|>", color=DIM, lw=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    ax.annotate(text, xy=(x, (y0 + y1) / 2), xytext=(-4, 0), textcoords="offset points",
                ha="right", va="center", rotation=90, fontsize=fs, color=DIM)


def _wall(ax, x_face, y0, y1, hw):
    """Hatched 'fixed support' strip sitting to the left of x_face."""
    ax.add_patch(Rectangle((x_face - hw, y0), hw, y1 - y0, fc="#f1f5f9", ec=WALL,
                           hatch="////", lw=0, zorder=1))
    ax.plot([x_face, x_face], [y0, y1], color=INK, lw=3.0, solid_capstyle="butt", zorder=4)


def _force(ax, tail, head):
    ax.annotate("", xy=head, xytext=tail, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=LOAD, lw=2.4,
                                shrinkA=0, shrinkB=0, mutation_scale=18))


def _load_label(ax, xy, text, offset, ha, va):
    ax.annotate(text, xy=xy, xytext=offset, textcoords="offset points", ha=ha, va=va,
                fontsize=11.5, fontweight="bold", color=LOAD, linespacing=1.25)


# ----------------------------------------------------------- plate / beam
def _draw_bar(L, W, hole_centres, D, P, theta_deg, fmt):
    """Shared drawing for the plate (1 hole) and beam (n holes): fixed left edge,
    load at the right edge, overall length / width dimensions."""
    th = np.deg2rad(theta_deg)
    a = 0.22 * L
    tip = (L + a * np.cos(th), a * np.sin(th))
    hw = 0.035 * L
    x_w = -hw - 0.045 * L
    gap = max(0.4 * W, 0.07 * L)
    y_dim = -W / 2 - gap

    xmin, xmax = x_w - 0.03 * L, tip[0] + 0.03 * L
    ymax = max(W / 2, tip[1]) + 0.04 * L
    ymin = min(y_dim, tip[1]) - 0.06 * L
    fig, ax = _canvas(xmin, xmax, ymin, ymax)

    ax.add_patch(Rectangle((0, -W / 2), L, W, fc=BODY, ec=EDGE, lw=1.6, zorder=2))
    for cx, cy in hole_centres:
        ax.add_patch(Circle((cx, cy), D / 2, fc="white", ec=HOLE, lw=1.7, zorder=3))
    _wall(ax, 0.0, -W / 2, W / 2, hw)
    _force(ax, (L, 0.0), tip)
    _load_label(ax, tip, f"P = {P:,.0f} N\nθ = {theta_deg:.1f}°", (9, 0), "left", "center")

    _hdim(ax, 0, L, y_dim, f"L = {L * 1e3:.0f} mm", ext0=-W / 2, ext1=-W / 2)
    _vdim(ax, x_w, -W / 2, W / 2, f"W = {W * 1e3:.1f} mm", ext0=-hw, ext1=-hw)
    return _encode(fig, fmt)


@lru_cache(maxsize=256)
def plate_schematic(L, W, D, x_off, y_off, P, theta_deg, fmt="svg"):
    return _draw_bar(L, W, [(L / 2 + x_off, y_off)], D, P, theta_deg, fmt)


@lru_cache(maxsize=256)
def beam_schematic(L, W, D, spacing, x_center, y_off, P, theta_deg, n_holes=2, fmt="svg"):
    half = (n_holes - 1) / 2.0 * spacing
    centres = [(x_center - half + i * spacing, y_off) for i in range(n_holes)]
    return _draw_bar(L, W, centres, D, P, theta_deg, fmt)


# ------------------------------------------------------------- rib bracket
@lru_cache(maxsize=256)
def rib_schematic(H, L_arm, t_wall, t_arm, rib_w, rib_h, a_frac, P, theta_deg,
                  mat_wall, mat_arm, mat_rib, patch_frac=0.15, fmt="svg"):
    y0 = H - t_arm
    th = np.deg2rad(theta_deg)
    hw = 0.045 * H
    x_tip = t_wall + L_arm
    xl = t_wall + a_frac * L_arm
    al = 0.30 * H
    tail = (xl - al * np.sin(th), H + al * np.cos(th))
    x_h = -hw - 0.06 * H
    y_dim = -0.12 * H

    xmin, xmax = x_h - 0.04 * H, x_tip + 0.10 * H
    ymin, ymax = y_dim - 0.09 * H, tail[1] + 0.13 * H
    fig, ax = _canvas(xmin, xmax, ymin, ymax)

    ax.add_patch(Polygon([(0, 0), (t_wall, 0), (t_wall, H), (0, H)], closed=True,
                         fc=MAT_COLORS[mat_wall], ec=EDGE, lw=1.3, zorder=2))
    ax.add_patch(Polygon([(t_wall, y0), (x_tip, y0), (x_tip, H), (t_wall, H)], closed=True,
                         fc=MAT_COLORS[mat_arm], ec=EDGE, lw=1.3, zorder=2))
    ax.add_patch(Polygon([(t_wall, y0), (t_wall + rib_w, y0), (t_wall, y0 - rib_h)], closed=True,
                         fc=MAT_COLORS[mat_rib], ec=EDGE, lw=1.3, zorder=3))
    _wall(ax, 0.0, 0.0, H, hw)

    c = patch_frac * L_arm  # load patch on the arm's top face
    ax.plot([xl - c / 2, xl + c / 2], [H, H], color=HOLE, lw=5, solid_capstyle="butt", zorder=5)
    _force(ax, tail, (xl, H))
    _load_label(ax, tail, f"P = {P:,.0f} N   θ = {theta_deg:.1f}°", (0, 6), "center", "bottom")

    _vdim(ax, x_h, 0, H, f"H = {H * 1e3:.0f} mm", ext0=-hw, ext1=-hw)
    _hdim(ax, t_wall, x_tip, y_dim, f"Arm L = {L_arm * 1e3:.0f} mm", ext0=0.0, ext1=y0)
    return _encode(fig, fmt)
