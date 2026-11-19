"""
Small 2D plane-stress FEA of the rib bracket, used only to draw the deformation contour.

Same set-up as the ANSYS model: back face of the wall fixed, force on a patch of the arm's top
edge, quadratic triangles. Geometry follows schematics.rib_schematic (origin = bottom-left of
the wall plate).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import triangle
from skfem import Basis, ElementTriP1, ElementTriP2, ElementVector, FacetBasis, LinearForm, MeshTri, condense, solve
from skfem.models.elasticity import lame_parameters, linear_elasticity

PATCH_FRAC = 0.15


def outline(H, L_arm, t_wall, t_arm, rib_w, rib_h, a_frac):
    """Bracket outline (counter-clockwise, metres) and the x-range of the load patch."""
    y0 = H - t_arm
    x_tip = t_wall + L_arm
    xl = t_wall + a_frac * L_arm
    c = PATCH_FRAC * L_arm
    pts = [(0, 0), (t_wall, 0), (t_wall, y0 - rib_h), (t_wall + rib_w, y0), (x_tip, y0),
           (x_tip, H), (xl + c / 2, H), (xl - c / 2, H), (0, H)]
    return np.array(pts, float), (xl - c / 2, xl + c / 2)


def _mesh(pts, h):
    n = len(pts)
    seg = [(i, (i + 1) % n) for i in range(n)]
    m = triangle.triangulate({"vertices": pts, "segments": seg}, f"pq30a{0.433 * h * h:.10f}")
    return MeshTri(m["vertices"].T.copy(), m["triangles"].T.copy())


@lru_cache(maxsize=64)
def solve_rib(H, L_arm, t_wall, t_arm, b, rib_w, rib_h, a_frac, Fx, Fy, E_gpa, nu, h=None):
    """Returns dict(mesh, u (2, n_nodes), max_mm). Lengths in m, forces in N."""
    pts, (xa, xb) = outline(H, L_arm, t_wall, t_arm, rib_w, rib_h, a_frac)
    h = h or min(t_wall, t_arm) / 1.6
    m = _mesh(pts, h)
    e = ElementVector(ElementTriP2())
    basis = Basis(m, e)

    E = E_gpa * 1e9
    lam, mu = lame_parameters(E, nu)
    lam = 2 * lam * mu / (lam + 2 * mu)                 # plane stress
    K = b * linear_elasticity(lam, mu).assemble(basis)

    patch = m.facets_satisfying(lambda x: (x[1] > H - 1e-9) & (x[0] >= xa - 1e-9) & (x[0] <= xb + 1e-9))
    fb = FacetBasis(m, e, facets=patch)
    c = xb - xa

    @LinearForm
    def load(v, w):
        return (Fx / c) * v.value[0] + (Fy / c) * v.value[1]

    f = load.assemble(fb)
    fixed = basis.get_dofs(m.facets_satisfying(lambda x: x[0] < 1e-9))
    u = solve(*condense(K, f, D=fixed))
    return dict(mesh=m, basis=basis, u=u, lam=lam, mu=mu)


def total_deformation(sol):
    """(refined mesh, |u| in mm at its nodes), max over the refined nodes."""
    (ux, bx), (uy, by) = sol["basis"].split(sol["u"])
    mr, vx = bx.refinterp(ux, 2)
    _, vy = by.refinterp(uy, 2)
    return mr, np.hypot(vx, vy) * 1e3, vx * 1e3, vy * 1e3


def von_mises(sol):
    """(refined mesh, equivalent stress in MPa), plane stress, smoothed by an L2 projection to P1."""
    b, u, lam, mu = sol["basis"], sol["u"], sol["lam"], sol["mu"]
    g = b.interpolate(u).grad                      # g[i][j] = d u_i / d x_j
    exx, eyy, exy = g[0][0], g[1][1], 0.5 * (g[0][1] + g[1][0])
    tr = exx + eyy
    sx, sy, txy = lam * tr + 2 * mu * exx, lam * tr + 2 * mu * eyy, 2 * mu * exy
    vm = np.sqrt(sx**2 - sx * sy + sy**2 + 3 * txy**2)
    b1 = b.with_element(ElementTriP1())
    mr, v = b1.refinterp(b1.project(vm), 2)
    return mr, v / 1e6


def solve_from_raw(raw, E_gpa, nu, h=None):
    th = np.deg2rad(raw["theta_deg"])
    return solve_rib(raw["H"], raw["L_arm"], raw["t_wall"], raw["t_arm"], raw["b"], raw["rib_w"],
                     raw["rib_h"], raw["a_frac"], raw["P"] * np.sin(th), -raw["P"] * np.cos(th),
                     float(E_gpa), float(nu), h)


def _nice(x):
    """Round a scale factor down to 1, 2 or 5 x 10^k."""
    k = 10 ** np.floor(np.log10(x))
    for m in (5, 2, 1):
        if m * k <= x:
            return m * k
    return k


def contour_png(raw, E_gpa, nu, kind="deformation", target=None, dpi=200):
    """Contour of total deformation (mm) or equivalent stress (MPa) on the deformed shape.
    If target is given the field is scaled so its maximum equals it (shape from the solver,
    magnitude from the surrogate). Returns (png_bytes, max_value, shape_exaggeration)."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    sol = solve_from_raw(raw, E_gpa, nu)
    mr, dfm, ux, uy = total_deformation(sol)
    scale = float(_nice(0.05 * raw["H"] * 1e3 / dfm.max())) if dfm.max() > 0 else 1.0
    if kind == "stress":
        _, f = von_mises(sol)
        label, unit = "Equivalent (von Mises) stress (MPa)", "MPa"
    else:
        f = dfm
        label, unit = "Total deformation (mm)", "mm"
    if target:
        f = f * (target / float(f.max()))
    fmax = float(f.max())
    x, y = mr.p[0] * 1e3, mr.p[1] * 1e3
    tri = Triangulation(x + scale * ux, y + scale * uy, mr.t.T)

    pts, _ = outline(raw["H"], raw["L_arm"], raw["t_wall"], raw["t_arm"], raw["rib_w"],
                     raw["rib_h"], raw["a_frac"])
    pts = pts * 1e3
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    levels = np.linspace(0, fmax, 11) if fmax > 0 else 11
    cs = ax.tricontourf(tri, f, levels=levels, cmap="jet")
    ax.plot(*np.vstack([pts, pts[:1]]).T, color="#475569", lw=0.9, ls="--", label="Undeformed")
    ax.plot([0, 0], [0, raw["H"] * 1e3], color="#0f172a", lw=3, solid_capstyle="butt")
    i = int(np.argmax(f))
    ax.plot(tri.x[i], tri.y[i], marker="o", ms=6, mfc="none", mec="#0f172a", mew=1.4)
    ax.annotate(f"max {fmax:.4g} {unit}", (tri.x[i], tri.y[i]), xytext=(-12, -26),
                textcoords="offset points", ha="right", fontsize=10, fontweight="bold",
                color="#0f172a", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5),
                arrowprops=dict(arrowstyle="-", color="#0f172a", lw=0.8))
    ax.set_aspect("equal")
    ax.margins(0.03)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cb = fig.colorbar(cs, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(label)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue(), fmax, scale
