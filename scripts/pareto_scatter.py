"""Paper-style model-comparison scatter: cost on x, quality on y, one point per
model, the Pareto frontier drawn, "better" corner marked. Shared by the Phase 1
(backbone) and Phase 4 (classifier) report builders.

    from pareto_scatter import pareto_scatter
    pareto_scatter(points, xlabel="size (MB)", ylabel="macro-F1", title="...",
                   out=Path("fig/scatter.png"), xlog=True)

`points`: list of dicts, each
    {"name": str, "x": float, "y": float,
     "size": float | None,     # bubble area key (e.g. VRAM); None -> uniform
     "group": str | None,      # marker shape key (e.g. "CPU"/"GPU")
     "star": bool}             # True -> gold star = the recommended pick

Lower x is always "better", higher y is always "better" (flip the metric before
calling if not). The frontier is the set of points that nothing beats on both.
"""

from __future__ import annotations

from pathlib import Path

_MARKERS = ["o", "^", "s", "D", "v"]
_STAR = "#f5a623"


def _frontier(pts):
    """Indices of non-dominated points (min x, max y)."""
    out = []
    for i, p in enumerate(pts):
        if not any(q["x"] <= p["x"] and q["y"] >= p["y"] and
                   (q["x"] < p["x"] or q["y"] > p["y"]) for q in pts):
            out.append(i)
    return sorted(out, key=lambda i: pts[i]["x"])


def pareto_scatter(points, *, xlabel, ylabel, title, out: Path,
                   xlog=False, note: str | None = None,
                   size_label: str = "size"):
    import matplotlib.pyplot as plt
    import numpy as np

    pts = list(points)
    groups = [p.get("group") for p in pts]
    uniq = [g for g in dict.fromkeys(groups) if g is not None]
    gmark = {g: _MARKERS[i % len(_MARKERS)] for i, g in enumerate(uniq)}

    sizes = [p.get("size") for p in pts]
    have_sz = any(s is not None for s in sizes)
    smax = max([s for s in sizes if s], default=1.0) or 1.0
    def area(s):
        if not have_sz:
            return 130.0
        return 60.0 + 900.0 * (s or 0.0) / smax

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    if xlog:
        ax.set_xscale("log")

    fi = _frontier(pts)
    fx = [pts[i]["x"] for i in fi]
    fy = [pts[i]["y"] for i in fi]
    ax.plot(fx, fy, "--", color="#1a7f37", lw=1.4, zorder=1, label="Pareto frontier")

    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    xhi = max(xs) / (min(xs) if xlog else 1)
    for i, p in enumerate(pts):
        m = gmark.get(p.get("group"), "o")
        ax.scatter(p["x"], p["y"], s=area(p.get("size")), marker=m,
                   facecolor=("#0969da" if i in fi else "#b9c4d0"),
                   edgecolor="#0b2038", lw=1.0, alpha=0.9, zorder=3)
        if p.get("star"):
            ax.scatter(p["x"], p["y"], s=260, marker="*", facecolor=_STAR,
                       edgecolor="#7a4d00", lw=0.8, zorder=4)
        # label left of points in the right third so it can't clip the frame
        frac = ((p["x"] / min(xs)) / xhi) if xlog else ((p["x"] - min(xs)) / (max(xs) - min(xs) + 1e-9))
        dx, ha = ((-9, "right") if frac > 0.62 else (8, "left"))
        odx, ody = p.get("label_offset", (0, 0))
        ax.annotate(p["name"], (p["x"], p["y"]), ha=ha,
                    textcoords="offset points", xytext=(dx + odx, 7 + ody),
                    fontsize=8.5, fontweight="bold" if p.get("star") else "normal")

    # "better" arrow into the top-left corner
    ax.annotate("better", xy=(0.04, 0.965), xycoords="axes fraction",
                xytext=(0.32, 0.72), textcoords="axes fraction",
                fontsize=10, color="#1a7f37", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#1a7f37", lw=1.6,
                                connectionstyle="arc3,rad=-0.15"))

    smin = min([s for s in sizes if s], default=0.0)
    if have_sz:
        ax.scatter([], [], s=0, label=f"bubble = {size_label}")
        for val in (smin, (smin + smax) / 2, smax):
            ax.scatter([], [], s=area(val), facecolor="#b9c4d0",
                       edgecolor="#0b2038", label=f"  {val:.3g}")
    for g in uniq:
        ax.scatter([], [], marker=gmark[g], facecolor="#0969da",
                   edgecolor="#0b2038", label=g)

    if xlog:
        ax.set_xlim(min(xs) / 1.8, max(xs) * 3.6)
    else:
        pad = 0.12 * (max(xs) - min(xs) + 1e-9)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ypad = 0.10 * (max(ys) - min(ys) + 1e-9)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad * 1.4)

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    if uniq or have_sz:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92,
                  labelspacing=0.7, borderpad=0.8)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, style="italic",
                 color="#555", wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def demo():
    import tempfile
    pts = [
        {"name": "A", "x": 1.0, "y": 0.9, "star": True},
        {"name": "B", "x": 10.0, "y": 0.92},
        {"name": "C", "x": 100.0, "y": 0.8},     # dominated by B
        {"name": "D", "x": 0.5, "y": 0.6},
    ]
    fi = _frontier(pts)
    assert set(fi) == {0, 1, 3}, fi                # C is dominated
    p = Path(tempfile.gettempdir()) / "pareto_demo.png"
    pareto_scatter(pts, xlabel="x", ylabel="y", title="demo", out=p, xlog=True)
    assert p.exists() and p.stat().st_size > 3000
    print("pareto_scatter.demo OK ->", p)


if __name__ == "__main__":
    demo()
