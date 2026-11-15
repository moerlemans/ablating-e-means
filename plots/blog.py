"""Figures and supplement of the write-up, from the committed results/sift_*.csv.

Figures 1-3 have two panels (figure 2 a third: with and without revival per budget), methods
introduced earlier drawn thin, controls dashed, runs without revival dotted:
  left   during training: test MSE along one 10-epoch run (before finalisation), with Lloyd's
         curve per iteration; relative to Lloyd after 25 iterations
  right  final result per epoch budget (one run per budget, B and alpha0 from the paper's laws
         at that budget); figure 1 relative to Lloyd after 25 iterations, figures 2-3 relative
         to e-means at the same budget and seed
Points with error bars are means with 95% confidence intervals over 5 seeds, dots single seeds; Lloyd's 5-seed mean is used as a
fixed reference.

blog/figures/fig0_headline.png        Lloyd, mini-batch k-means, e-means and centroid EMA++ per budget (social media)
blog/figures/fig1_centroid_ema.png    Lloyd, mini-batch k-means, e-means, centroid EMA
blog/figures/fig2_count_scaled.png    + centroid EMA+
blog/figures/fig3_no_revival.png      mini-batch k-means, e-means and centroid EMA+ without revival
blog/figures/fig4_error_revival.png   centroid EMA++, e-means + error revival and the controls
blog/supplement.md                    all numbers behind the post
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from emeans.methods import ABLATIONS, CONTROLS, METHODS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BLOG = ROOT / "blog"

E_MEANS_CTRL, EMA_PLUS_CTRL = CONTROLS
NO_REVIVAL = [
    "mini-batch k-means, no revival",
    "e-means, no revival",
    "centroid EMA+, no revival",
]  # extra runs in experiments/sift.py
# the colour follows the update rule in every figure (e-means blue, centroid EMA+ green, also with
# error revival); the marker shows the revival rule where it varies
COLORS = {
    "mini-batch k-means": "#3b3b3b",
    "mini-batch k-means, no revival": "#3b3b3b",
    "e-means": "#2a78d6",
    "centroid EMA": "#eb6834",
    "centroid EMA+": "#1baf7a",
    "centroid EMA++": "#1baf7a",
    "e-means + error revival": "#2a78d6",
    E_MEANS_CTRL: "#2a78d6",
    EMA_PLUS_CTRL: "#1baf7a",
    "e-means, no revival": "#2a78d6",
    "centroid EMA+, no revival": "#1baf7a",
}
LABELS = {
    "centroid EMA+": "centroid EMA+ (count-scaled step, warm-up)",
    "centroid EMA++": "centroid EMA++ (count-scaled step, warm-up, error revival)",
}
LLOYD = "#6250d6"
INK_2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "font.size": 10,
    }
)


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    h = st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def curve(df, x, y):
    rows = [(k, *mean_ci(g[y])) for k, g in df.groupby(x)]
    return pd.DataFrame(rows, columns=[x, "mean", "lo", "hi"])


def paired(df, by, base, ref):
    """Test MSE of df minus e-means (`base`) at the same `by` keys, in points of Lloyd@25."""
    d = df.set_index(by).test_mse - base
    return (100 * d / ref).rename("diff").reset_index()


def load():
    lloyd = pd.read_csv(RESULTS / "sift_lloyd.csv")
    budgets = pd.read_csv(RESULTS / "sift_budgets.csv")
    curves = pd.read_csv(RESULTS / "sift_curves.csv")
    for df, by in ((lloyd, ["iter"]), (budgets, ["method", "epochs"]), (curves, ["method", "epoch"])):
        n = df.groupby(by).agg(configs=("config", "nunique"), seeds=("seed", "nunique"))
        bad = n[(n.configs != 1) | (n.seeds != 5)]
        if len(bad):
            raise ValueError(f"results mix configurations or miss seeds:\n{bad}")
    ref = lloyd[lloyd.iter == 25].test_mse.mean()
    for df in (lloyd, budgets, curves):
        df["rel"] = 100 * (df.test_mse / ref - 1)
    base_b = budgets[budgets.method == "e-means"].set_index(["epochs", "seed"]).test_mse
    base_c = curves[curves.method == "e-means"].set_index(["epoch", "seed"]).test_mse
    budget_diff = {m: paired(g, ["epochs", "seed"], base_b, ref) for m, g in budgets.groupby("method")}
    curve_diff = {m: paired(g, ["epoch", "seed"], base_c, ref) for m, g in curves.groupby("method")}
    return lloyd, budgets, curves, ref, budget_diff, curve_diff


def save(fig, name):
    path = BLOG / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


BUDGETS = [1, 2, 3, 5, 10]
BUDGET_LABEL = "epoch budget (one run per budget)"
LLOYD_REF = ("Lloyd, 25 iterations", INK_2)
EMEANS_REF = ("e-means", "#2a78d6")
VS_LLOYD = "test MSE relative to Lloyd, 25 iterations (%)"
VS_EMEANS = "test MSE minus e-means (points)"


def style(method, new, override=None):
    """Colour, marker, line style and label of a method. Filled markers for the methods a figure
    introduces, hollow for methods from earlier figures; a triangle for runs without revival.
    `override` replaces any of these per figure."""
    color = COLORS[method]
    st_ = {
        "color": color,
        "marker": "v" if method in NO_REVIVAL else "o",
        "mfc": color if method in new else SURFACE,
        "alpha": 1 if method in new else 0.75,
        "linestyle": "-",
        "label": LABELS.get(method, method),
    }
    st_.update((override or {}).get(method, {}))
    return st_


def points(ax, df, x, y, xs, st_, seeds=0.0, line=True):
    """Mean and 95% CI per x as a point with an error bar at positions xs, joined by a faint line,
    plus the single seeds shifted `seeds` to the right of their bar (0: no seeds)."""
    c = curve(df, x, y)
    pos = xs(c[x].to_numpy())
    if line:
        ax.plot(pos, c["mean"], color=st_["color"], linewidth=1.2, alpha=0.35, linestyle=st_["linestyle"], zorder=2)
    ax.errorbar(
        pos,
        c["mean"],
        yerr=[c["mean"] - c.lo, c.hi - c["mean"]],
        fmt=st_["marker"],
        color=st_["color"],
        mfc=st_["mfc"],
        mec=st_["color"],
        ms=5.5,
        elinewidth=1.4,
        capsize=2.5,
        alpha=st_["alpha"],
        label=st_["label"],
        zorder=3,
    )
    if seeds:
        ax.scatter(xs(df[x].to_numpy()) + seeds, df[y], s=9, color=st_["color"], alpha=0.45, linewidths=0, zorder=2)


def reference(ax, ref, label=True):
    """Dashed zero line, labelled at the left edge just below the line."""
    ax.axhline(0, color=ref[1], linewidth=1, linestyle="--", zorder=1)
    if label:
        ax.annotate(ref[0], (0.005, 0), xycoords=("axes fraction", "data"), xytext=(2, -3),
                    textcoords="offset points", color=ref[1], fontsize=9, ha="left", va="top")


def budget_panel(ax, frames, methods, new, y, ylabel, title, ref, override=None):
    """Per budget a group of points, one per method, side by side, each method joined across budgets."""
    width = 0.78
    step = width / len(methods)
    index = {b: i for i, b in enumerate(BUDGETS)}
    for j, m in enumerate(methods):
        off = (j - (len(methods) - 1) / 2) * step
        xs = lambda v, off=off: np.array([index[b] for b in v]) + off  # noqa: E731
        points(ax, frames[m], "epochs", y, xs, style(m, new, override), seeds=min(0.05, step / 3))
    for i in range(len(BUDGETS) - 1):
        ax.axvline(i + 0.5, color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(BUDGETS)), [str(b) for b in BUDGETS])
    ax.set_xlim(-0.5, len(BUDGETS) - 0.5)
    reference(ax, ref)
    ax.set_xlabel(BUDGET_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)


def log_axis(ax, ticks, xlabel):
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.9, ticks[-1] * 1.1)
    ax.set_xlabel(xlabel)


def band(ax, df, x, y, st_, thin=False):
    """Mean line with a 95% CI band (thin: line only, for methods from earlier figures)."""
    c = curve(df, x, y)
    if thin:
        ax.plot(c[x], c["mean"], color=st_["color"], linewidth=1, linestyle=st_["linestyle"], label=st_["label"])
        return
    ax.fill_between(c[x], c.lo, c.hi, color=st_["color"], alpha=0.15, linewidth=0)
    ax.plot(c[x], c["mean"], color=st_["color"], linestyle=st_["linestyle"], marker=st_["marker"],
            markersize=3.5, mfc=st_["color"], label=st_["label"])


def lloyd_line(ax, lloyd, label="Lloyd (1 iteration = 1 epoch)"):
    band(ax, lloyd[lloyd.iter >= 1], "iter", "rel", {"color": LLOYD, "linestyle": "-", "marker": None, "label": label})


def training_panel(ax, lloyd, curves, new, old, override=None):
    """Test MSE after every epoch of one 10-epoch run, with Lloyd's curve per iteration."""
    lloyd_line(ax, lloyd)
    for m in old + new:
        band(ax, curves[curves.method == m], "epoch", "rel", style(m, new, override), thin=m not in new)
    log_axis(ax, [1, 2, 3, 5, 10, 25], "epoch (log scale)")
    reference(ax, LLOYD_REF)
    ax.set_ylabel(VS_LLOYD)
    ax.set_title("During training (one 10-epoch run, before finalisation)")
    ax.legend(loc="upper right")


def two_panel(title):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True, width_ratios=[1, 1.15])
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return fig, a, b


TRAIN_NOTE = "left: mean and 95% CI over 5 seeds"
BUDGET_NOTE = "right: points mean, bars 95% CI over 5 seeds, dots single seeds"


def figures(lloyd, budgets, curves, budget_diff):
    per_method = {m: g for m, g in budgets.groupby("method")}
    per_budget = "Final result per epoch budget"
    vs_emeans = f"{per_budget}, relative to e-means"
    notes = f"({TRAIN_NOTE}; {BUDGET_NOTE})"

    fig, a, b = two_panel(f"Mini-batch k-means, e-means and a plain centroid EMA {notes}")
    new = ["mini-batch k-means", "e-means", "centroid EMA"]
    training_panel(a, lloyd, curves, new, [])
    budget_panel(b, per_method, new, new, "rel", VS_LLOYD, per_budget, LLOYD_REF)
    save(fig, "fig1_centroid_ema.png")

    fig, a, b = two_panel(f"Centroid EMA+: a count-scaled step and a warm-up {notes}")
    training_panel(a, lloyd, curves, ["centroid EMA+"], ["e-means", "centroid EMA"])
    budget_panel(b, budget_diff, ["centroid EMA", "centroid EMA+"], ["centroid EMA+"], "diff", VS_EMEANS, vs_emeans,
                 EMEANS_REF)
    save(fig, "fig2_count_scaled.png")

    # with revival on the left of the legend (e-means is the zero line), without on the right
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    fig.suptitle(f"With and without revival ({BUDGET_NOTE[7:]})", x=0.01, ha="left", fontsize=12)
    with_revival = ["mini-batch k-means", "centroid EMA+"]
    override = {m: {"label": f"{m}, with revival", "mfc": SURFACE, "alpha": 1} for m in with_revival}
    budget_panel(ax, budget_diff, [*with_revival, *NO_REVIVAL], NO_REVIVAL, "diff", VS_EMEANS,
                 f"{per_budget}, relative to e-means with revival", EMEANS_REF, override)
    handles, labels = ax.get_legend_handles_labels()
    ref = plt.Line2D([], [], color=COLORS["e-means"], linewidth=1, linestyle="--")
    ax.legend([ref, *handles], ["e-means, with revival (reference)", *labels], loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2)
    save(fig, "fig3_no_revival.png")

    # the four combinations of update (e-means, centroid EMA+) and revival rule (count, error),
    # all checking for dead clusters every batch; colour = update, marker and line = revival rule
    fig, a, b = two_panel(f"Count revival versus error revival, both every batch {notes}")
    four = [E_MEANS_CTRL, EMA_PLUS_CTRL, "e-means + error revival", "centroid EMA++"]
    override = {
        E_MEANS_CTRL: {"label": "e-means, count revival"},
        EMA_PLUS_CTRL: {"label": "centroid EMA+, count revival"},
        "e-means + error revival": {"label": "e-means, error revival", "marker": "s", "linestyle": "--"},
        "centroid EMA++": {"label": "centroid EMA++ (= centroid EMA+, error revival)", "marker": "s",
                           "linestyle": "--"},
    }
    training_panel(a, lloyd, curves, four, [], override)
    budget_panel(b, per_method, four, four, "rel", VS_LLOYD, per_budget, LLOYD_REF, override)
    save(fig, "fig4_error_revival.png")


def headline(lloyd, budgets):
    """One figure for social media: test MSE per epoch budget for Lloyd, e-means and centroid EMA++."""
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    lloyd_line(ax, lloyd, "Lloyd (1 iteration = 1 pass over the data)")
    new = ["mini-batch k-means", "e-means", "centroid EMA++"]
    for j, m in enumerate(new):
        nudge = 10 ** ((j - 1) * 0.015)
        xs = lambda v, f=nudge: v * f  # noqa: E731
        points(ax, budgets[budgets.method == m], "epochs", "rel", xs, style(m, new))
    log_axis(ax, [1, 2, 3, 5, 10, 25], "epoch budget: number of passes over the data (log scale)")
    reference(ax, LLOYD_REF)
    ax.set_ylabel("test MSE relative to Lloyd after 25 iterations (%)")
    ax.set_title("k-means on SIFT (200k vectors, k = 2048); points: mean, bars: 95% CI over 5 seeds")
    ax.legend(loc="upper right")
    fig.suptitle(
        "e-means and centroid EMA++ match 25 iterations of Lloyd in about 3 passes",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    save(fig, "fig0_headline.png")


def table(df):
    def cell(v):
        return f"{v:g}" if isinstance(v, float) and float(v).is_integer() else str(v)

    header = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return (
        header + "\n".join("| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)) + "\n"
    )


def fmt(values):
    m, lo, hi = mean_ci(values)
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def minus(budgets, m, T, col, ref):
    """Column `col` of method m minus e-means at budget T, paired by seed, in points of `ref`."""
    by = budgets[budgets.epochs == T].set_index(["method", "seed"])[col]
    return 100 * (by.loc[m] - by.loc["e-means"]) / ref


def budget_rows(budgets, names, ref):
    """Per method and budget: test MSE vs Lloyd, and test, train and before-finalisation test MSE
    minus e-means (paired by seed), plus cluster health."""
    rows = []
    for m in names:
        for T, g in budgets[budgets.method == m].groupby("epochs"):
            other = m != "e-means"
            rows.append(
                {
                    "method": m,
                    "budget": T,
                    "test vs Lloyd@25 (%)": fmt(g.rel),
                    "test minus e-means": fmt(minus(budgets, m, T, "test_mse", ref)) if other else "",
                    "train minus e-means": fmt(minus(budgets, m, T, "train_mse", ref)) if other else "",
                    "test minus e-means, before finalisation": fmt(
                        minus(budgets, m, T, "test_mse_before_finalisation", ref)
                    )
                    if other
                    else "",
                    "dead (<10 pts)": f"{g.dead.mean():.1f}",
                    "imbalance": f"{g.imbalance.mean():.3f}",
                    "revived": f"{g.revived.mean():.0f}",
                    "after epoch 1": f"{g.revived_after_epoch_1.mean():.0f}",
                    "re-death": f"{g.redeath_rate.mean():.3f}",
                }
            )
    return pd.DataFrame(rows)


def overlap_rows(overlap):
    """Clusters that newly fall below each rule's threshold, pooled over passes and seeds, for the
    first epoch and the rest of the run: how often the other rule agrees, and the mean rank
    correlation of the two donor weights."""
    rows = []
    for (m, T), g in overlap.groupby(["method", "epochs"]):
        for window, w in [
            ("first epoch", g[g.batch <= g.batches_per_epoch]),
            ("after the first epoch", g[g.batch > g.batches_per_epoch]),
        ]:
            if w.empty:
                continue
            e, c = w.new_by_error.sum(), w.new_by_count.sum()
            e_also, c_also = w.new_by_error_also_count.sum(), w.new_by_count_also_error.sum()
            rows.append(
                {
                    "run": m,
                    "budget": T,
                    "passes": window,
                    "newly dead by error": int(e),
                    "of which also dead by count": f"{e_also / e:.2f}" if e else "",
                    "newly dead by count": int(c),
                    "of which also dead by error": f"{c_also / c:.2f}" if c else "",
                    "donor rank corr.": f"{w.donor_rank_corr.mean():.2f}",
                }
            )
    return pd.DataFrame(rows)


def supplement(lloyd, budgets, curves, ref, budget_diff, curve_diff):
    summary = pd.read_csv(RESULTS / "sift_lloyd_summary.csv")
    tune = pd.read_csv(RESULTS / "sift_tune.csv")
    overlap = pd.read_csv(RESULTS / "sift_overlap.csv")
    s = [
        "# Supplement",
        "",
        "Means with 95% confidence intervals over seeds 0-4, test MSE on the SIFT queries relative to Lloyd after 25 "
        f"iterations (the mean over the 5 seeds, {ref:.1f}, used as a fixed reference), unless stated otherwise. "
        "Differences are paired by seed and given in points of that reference. Runs with the same seed share the "
        "initial centroids and the order of the batches.",
        "",
    ]

    lloyd25 = lloyd[lloyd.iter == 25].set_index("seed").test_mse
    rows = [
        {
            "epoch budget": T,
            "e-means minus Lloyd@25, same seed (points)": fmt(100 * (g.set_index("seed").test_mse - lloyd25) / ref),
        }
        for T, g in budgets[budgets.method == "e-means"].groupby("epochs")
    ]
    s += [
        "## Lloyd",
        "",
        f"Lloyd after 25 iterations refilled {summary.empty_splits_25.mean():.1f} empty clusters per run on average. "
        f"To convergence it takes {summary.converged_iters.mean():.0f} iterations on average "
        f"({summary.converged_iters.min()}-{summary.converged_iters.max()}) and ends at "
        f"{fmt(100 * (summary.converged_test_mse / ref - 1))}%. e-means minus Lloyd after 25 iterations, per seed "
        "(this includes Lloyd's own seed-to-seed spread):",
        "",
        table(pd.DataFrame(rows)),
    ]

    laws = budgets.groupby("epochs")[["batch_size", "alpha0"]].first().reset_index()
    laws["batches per epoch"] = 200_000 // laws.batch_size
    laws["alpha0"] = laws.alpha0.round(3)
    s += [
        "## Final result per epoch budget",
        "",
        "Batch size and peak step from the paper's laws at each budget:",
        "",
        table(laws),
        "",
        table(budget_rows(budgets, list(METHODS) + list(CONTROLS), ref)),
        "",
        "### The two changes of centroid EMA+ separately",
        "",
        table(budget_rows(budgets, ["centroid EMA"] + list(ABLATIONS) + ["centroid EMA+"], ref)),
    ]

    rows = []
    for m in list(METHODS) + list(CONTROLS):
        for e, g in curves[curves.method == m].groupby("epoch"):
            d = curve_diff[m][curve_diff[m].epoch == e]["diff"]
            rows.append(
                {
                    "method": m,
                    "epoch": e,
                    "test vs Lloyd@25 (%)": fmt(g.rel),
                    "minus e-means (points)": "" if m == "e-means" else fmt(d),
                }
            )
    ll = curve(lloyd[lloyd.iter >= 1], "iter", "rel")
    s += [
        "## During training",
        "",
        "One 10-epoch run per method and seed (batch size 32768 and peak step 0.914 from the laws at 10 epochs, so 6 "
        "batches per epoch), test MSE after every epoch, before finalisation.",
        "",
        table(pd.DataFrame(rows)),
        "",
        "Lloyd per iteration: "
        + ", ".join(f"{int(r.iter)}: {r['mean']:+.2f}%" for _, r in ll[ll.iter <= 10].iterrows()),
        "",
    ]

    rows = []
    for T, gt in tune.groupby("epochs"):
        base = gt[(gt.method == "e-means") & (gt.exponent == 0)].val_mse.mean()
        for m in METHODS:
            rel = 100 * (gt[gt.method == m].groupby("exponent").val_mse.mean() / base - 1)
            best = rel.idxmin()
            rows.append(
                {
                    "budget": T,
                    "method": m,
                    "at the law": f"{rel[0.0]:+.3f}",
                    "best alpha0 / law": f"{2 ** best:.2f}",
                    "at the best": f"{rel[best]:+.3f}",
                }
            )
    s += [
        "## Step size check",
        "",
        "Validation MSE (10k held-out base vectors, mean over seeds 100-102) for alpha0 = law x 2^j, "
        "j in {-2, -1.5, ..., 2}, in % relative to e-means at the law value of the same budget. Values with "
        "alpha0 >= 1 are skipped. The best value is picked on the same 3 seeds, so it is optimistic.",
        "",
        table(pd.DataFrame(rows)),
    ]

    s += [
        "## Which clusters the two revival rules call dead",
        "",
        "At every revival pass of e-means (count revival every 5 batches, the error EMA tracked on the side) and of "
        "e-means + error revival (every batch): the clusters that newly fall below 10% of the mean error or of the "
        "mean count (a cluster that stays below is counted once), and whether the other rule calls them dead at the "
        "same pass; summed over passes and seeds 0-4. Plus the Spearman correlation of the two donor weights over "
        "the clusters neither rule calls dead (mean over passes).",
        "",
        table(overlap_rows(overlap)),
    ]

    path = BLOG / "supplement.md"
    path.write_text("\n".join(s))
    print(f"wrote {path}")


def main():
    lloyd, budgets, curves, ref, budget_diff, curve_diff = load()
    headline(lloyd, budgets)
    figures(lloyd, budgets, curves, budget_diff)
    supplement(lloyd, budgets, curves, ref, budget_diff, curve_diff)


if __name__ == "__main__":
    main()
