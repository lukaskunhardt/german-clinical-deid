"""Draw quality against model size for every model in a results file, with the Pareto front.

  python plot_pareto.py results/dev_comparison.json --metric char_f1 --out pareto_grascco_dev
  python plot_pareto.py results/dev_comparison.json --metric leak_rate --lower-is-better --out pareto_leaks_dev

Results file (JSON):
  {"title": "...", "caption": "...", "metrics": {"char_f1": "Character F1, GraSCCo dev (22 letters)", ...},
   "models": [{"name": "...", "params": 335737345, "released": true,
               "rows": [{"setting": "default" | "matched threshold", "char_f1": 0.97, ...}]}]}
Each row is one point. Two rows of the same model (its default and a threshold chosen by the same rule as mine) are joined by a line.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MINE, OTHER, INK, MUTED = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"


def front(points, lower):
    """Pareto front: no other point is both smaller and better."""
    best, out = None, []
    for x, y in sorted(points):
        if best is None or (y < best if lower else y > best):
            out.append((x, y))
            best = y
    return out


def place_labels(ax, labels):
    """Label each model once, at its better point; move a label down when it would cover one placed before."""
    fig = ax.figure
    fig.canvas.draw()
    taken = []
    for x, y, text, mine in sorted(labels, key=lambda t: (-t[1], t[0])):
        dy = 4
        while True:
            t = ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", fontsize=11, color=INK,
                            fontweight="bold" if mine else "normal")
            box = t.get_window_extent(fig.canvas.get_renderer()).expanded(1.0, 1.15)
            if not any(box.overlaps(b) for b in taken) or dy < -60:
                taken.append(box)
                break
            t.remove()
            dy -= 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--metric", default="char_f1")
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--out", default="pareto")
    a = ap.parse_args()
    res = json.load(open(a.results, encoding="utf-8"))

    plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    ax.set_xscale("log")
    pts, labels = [], []
    for m in res["models"]:
        rows = [r for r in m["rows"] if r.get(a.metric) is not None]
        if not rows:
            continue
        x, ys = m["params"] / 1e6, [r[a.metric] for r in rows]
        color = MINE if m.get("released") else OTHER
        if len(ys) > 1:
            ax.plot([x] * len(ys), ys, color=color, lw=2, zorder=1)
        for r, y in zip(rows, ys):
            filled = r["setting"] != "default"
            ax.scatter([x], [y], s=110 if m.get("released") else 70, zorder=3, linewidths=2, edgecolors=color,
                       facecolors=color if filled else "white")
            pts.append((x, y))
        best = min(ys) if a.lower_is_better else max(ys)
        labels.append((x, best, m["name"] if m.get("released") else m["name"].split("/")[-1], m.get("released")))
    place_labels(ax, labels)
    f = front(pts, a.lower_is_better)
    fx, fy = [p[0] for p in f], [p[1] for p in f]
    ax.step(fx + [ax.get_xlim()[1]], fy + [fy[-1]], where="post", color=MINE, lw=1.5, ls="--", zorder=2, label="Pareto front")
    ax.scatter([], [], s=70, facecolors=OTHER, edgecolors=OTHER, label="threshold chosen on development data (same rule for every model)")
    ax.scatter([], [], s=70, facecolors="white", edgecolors=OTHER, linewidths=2, label="default setting of the model")
    ax.scatter([], [], s=110, facecolors=MINE, edgecolors=MINE, label="released model")
    ax.set_xlabel("Model size (million parameters, log scale)")
    ax.set_ylabel(res["metrics"][a.metric] + (" (lower is better)" if a.lower_is_better else " (higher is better)"))
    ax.set_title(res["title"], fontsize=18, color=INK, loc="left")
    ax.grid(True, color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=12)
    fig.text(0.01, 0.01, res["caption"], fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(f"{a.out}.{ext}", facecolor="white")
    print("wrote", a.out + ".png", a.out + ".svg")


if __name__ == "__main__":
    main()
