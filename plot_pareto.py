"""Draw quality against model size for every model in a results file, with the Pareto front.

  python plot_pareto.py results/test_comparison.json --metric leak_rate --lower-is-better --out pareto_test_mdeid_leak_rate

Results file (JSON):
  {"title": "...", "caption": "...", "metrics": {"char_f1": "Character F1, GraSCCo dev (22 letters)", ...},
   "models": [{"name": "...", "params": 335737345, "released": true,
               "rows": [{"setting": "default" | "matched threshold", "char_f1": 0.97, ...}]}]}

Each model is one point: the better of its rows (its default or a threshold chosen by the same rule as mine). Only the released
model and the strongest other models get a name; every model is in the results table.
"""
import argparse
import json

import matplotlib
import matplotlib.text
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MINE, FRONT, OTHER, INK, MUTED = "#1f6fd1", "#1f6fd1", "#a3a29d", "#0b0b0b", "#52514e"
N_STRONGEST = 4


def front(points, lower):
    """Pareto front: no other model is both smaller and better."""
    best, out = None, []
    for x, y, name in sorted(points):
        if best is None or (y < best if lower else y > best):
            out.append((x, y, name))
            best = y
    return out


def place_labels(ax, items, dots, line):
    """Put each name next to its point where it covers no other name, no point and no part of the front line."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dot_boxes = [ax.transData.transform((x, y)) for x, y in dots]
    # the front line, sampled densely in screen coordinates
    pts = [ax.transData.transform(p) for p in line]
    samples = [(x0 + (x1 - x0) * k / 60, y0 + (y1 - y0) * k / 60) for (x0, y0), (x1, y1) in zip(pts, pts[1:]) for k in range(61)]
    taken = []
    for x, y, text, bold in items:
        candidates = [(12, 0, "left", "center"), (-12, 0, "right", "center")]
        for d in (16, 30, 46):  # farther away only when nearer places are taken
            candidates += [(12, d, "left", "bottom"), (12, -d, "left", "top"), (-12, d, "right", "bottom"),
                           (-12, -d, "right", "top"), (0, d, "center", "bottom"), (0, -d, "center", "top")]
        for dx, dy, ha, va in candidates:
            # measure a plain text at the candidate place first; the annotation with its pointer comes only when the place is free
            px, py = ax.transData.transform((x, y))
            k = fig.dpi / 72
            probe = ax.text(px + dx * k, py + dy * k, text, transform=None, ha=ha, va=va, fontsize=14,
                            fontweight="bold" if bold else "normal")
            box = probe.get_window_extent(r).expanded(1.04, 1.25)
            probe.remove()
            inside_axes = ax.get_window_extent(r).contains(box.x0, box.y0) and ax.get_window_extent(r).contains(box.x1, box.y1)
            hits_dot = any(box.x0 - 6 < px < box.x1 + 6 and box.y0 - 6 < py < box.y1 + 6 for px, py in dot_boxes)
            hits_line = any(box.x0 - 2 < px < box.x1 + 2 and box.y0 - 2 < py < box.y1 + 2 for px, py in samples)
            if inside_axes and not hits_dot and not hits_line and not any(box.overlaps(b) for b in taken):
                taken.append(box)
                # a label beside the point needs no pointer; one above or below gets a thin line to its point
                arrow = None if dy == 0 else dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=5)
                ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=14,
                            color=MINE if bold else INK, fontweight="bold" if bold else "normal", arrowprops=arrow)
                break
        else:
            raise SystemExit(f"no free place for the label {text!r}; label fewer models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--metric", default="char_f1")
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--percent", action="store_true", help="show the metric as a percentage")
    ap.add_argument("--out", default="pareto")
    a = ap.parse_args()
    res = json.load(open(a.results, encoding="utf-8"))
    pick = min if a.lower_is_better else max
    scale = 100 if a.percent else 1

    models = []
    for m in res["models"]:
        ys = [r[a.metric] for r in m["rows"] if r.get(a.metric) is not None]
        if ys:
            models.append((m["params"] / 1e6, pick(ys) * scale, m["name"].split("/")[-1], bool(m.get("released"))))

    plt.rcParams.update({"font.size": 16, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    ax.set_xscale("log")
    xs = [m[0] for m in models]
    ys = [m[1] for m in models]
    ax.set_xlim(min(xs) / 1.25, max(xs) * 1.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    pad = (max(ys) - min(ys)) * 0.08
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    f = front([(x, y, n) for x, y, n, _ in models], a.lower_is_better)
    fx, fy = [p[0] for p in f], [p[1] for p in f]
    ax.step(fx + [ax.get_xlim()[1]], fy + [fy[-1]], where="post", color=FRONT, lw=1.6, ls="--", zorder=1)
    line = []
    for i in range(len(fx)):
        x_next = fx[i + 1] if i + 1 < len(fx) else ax.get_xlim()[1]
        line += [(fx[i], fy[i]), (x_next, fy[i])]
        if i + 1 < len(fx):
            line.append((fx[i + 1], fy[i + 1]))
    for x, y, _, mine in models:
        ax.scatter([x], [y], s=220 if mine else 90, color=MINE if mine else OTHER, zorder=3,
                   edgecolors="white", linewidths=1.5)

    others = sorted([m for m in models if not m[3]], key=lambda m: m[1], reverse=not a.lower_is_better)
    named = {m[2] for m in models if m[3]} | {m[2] for m in others[:N_STRONGEST]}
    items = [(x, y, n, mine) for x, y, n, mine in sorted(models, key=lambda m: (not m[3], m[1])) if n in named]
    ax.set_xlabel("Model size (million parameters, log scale)")
    ax.set_ylabel(res["metrics"][a.metric].split(",")[0] + (" (%)" if a.percent else "") + (", lower is better" if a.lower_is_better else ", higher is better"))
    ax.set_title(res["metrics"][a.metric], fontsize=20, color=INK, loc="left", pad=14)
    ax.grid(True, color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.2)
    place_labels(ax, items, [(m[0], m[1]) for m in models], line)

    handles = [Line2D([], [], marker="o", ls="", ms=14, color=MINE, label="my models" if sum(m[3] for m in models) > 1 else "my model"),
               Line2D([], [], marker="o", ls="", ms=9, color=OTHER, label="other open model (better of its two settings)"),
               Line2D([], [], color=FRONT, ls="--", lw=1.6, label="Pareto front: no smaller model is better")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.08, 0.065), ncol=3, frameon=False, fontsize=14)
    fig.text(0.08, 0.02, res["caption"], fontsize=13, color=MUTED)
    for ext in ("png", "svg"):
        fig.savefig(f"{a.out}.{ext}", facecolor="white")
    print("wrote", a.out + ".png", a.out + ".svg")


if __name__ == "__main__":
    main()
