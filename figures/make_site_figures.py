"""Generate the model-results figures (06-12) used on the project page.

These charts visualize results that already exist in the project notebooks
(https://github.com/Poudel-Sanskriti/Sleep_Doctor). Nothing here re-fits a
model or reads the dataset: every number below is transcribed from the
executed notebook outputs and from `models.json`, with its provenance noted
in a comment. That keeps the site repo small and makes each chart auditable
against the source it came from.

Figures 01-05 come from `notebooks/03_visualization.ipynb` in the project
repo and are copied in unchanged -- this script does not touch them.

    python figures/make_site_figures.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = "figures"

# --- palette (colorblind-safe: no meaning is ever carried by red vs green) ---
BLUE = "#3987e5"      # primary marks
NAVY = "#1a4c8b"      # dark accent / second family
RED = "#e66767"       # negative or below-average pole
GRAY = "#9aa0a6"      # neutral / baseline
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e4de"

# --- one style for every chart: recessive grid, no chart junk ---
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,          # grid stays BEHIND the data
    "axes.labelcolor": INK, "axes.edgecolor": INK_SOFT,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "font.size": 10,
})


def note(fig, text, dy=-0.045):
    """Small provenance/caption line under the plot."""
    fig.text(0.0, dy, text, ha="left", va="top", fontsize=8.5,
             color=INK_SOFT, wrap=True)


def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 06 -- final test-set accuracy, one visit per model (notebooks 04 and 4.5)
# ---------------------------------------------------------------------------
def model_scorecard():
    # worst to best; baseline is the majority-class rule (always guess Medium)
    labels = ["Baseline\n(always guess Medium)",
              "Primary\nLinear Regression",
              "Primary\nRandom Forest (d=8)",
              "Extension\nLinear Regression",
              "Extension\nGradient Boosting"]
    values = [44.5, 61.2, 61.3, 68.3, 68.8]
    colors = [GRAY, BLUE, BLUE, NAVY, NAVY]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    y = np.arange(len(values))
    bars = ax.barh(y, values, color=colors, height=0.66,
                   edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, color=INK)

    ax.axvline(44.5, color=INK_SOFT, linestyle="--", linewidth=1.2, zorder=3)
    ax.text(45.6, 4.42, "baseline 44.5%", fontsize=9, color=INK_SOFT,
            va="center")

    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("Held-out test accuracy (%)")
    ax.set_xlim(0, 80)
    ax.grid(axis="y", visible=False)
    ax.set_title("Both proposed models clear the baseline by ~17 points —\n"
                 "then tie with each other", loc="left")
    note(fig, "Test-set accuracy, each final model evaluated exactly once. "
              "Primary models use age, stress and activity; extension models "
              "add 13 lifestyle/context features.")
    save(fig, "06_model_scorecard.png")


# ---------------------------------------------------------------------------
# 07 -- permutation importance, Path A (notebook 04)
# ---------------------------------------------------------------------------
def permutation_importance():
    labels = ["stress_score", "age", "exercise_day", "steps_that_day"]
    values = [0.8248, 0.0010, 0.0007, 0.0000]

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(values))[::-1]          # largest at top
    bars = ax.barh(y, values, color=[BLUE, GRAY, GRAY, GRAY], height=0.6,
                   edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values],
                 padding=4, fontsize=10, color=INK)

    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlabel("Drop in R² when the feature is shuffled")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_title("Shuffle stress and the model collapses —\n"
                 "shuffle anything else and nothing happens", loc="left")
    note(fig, "Permutation importance on Path A (Linear Regression). "
              "scikit-learn's default impurity importances misleadingly "
              "ranked step count first, which is why permutation importance "
              "is the measure reported here.")
    save(fig, "07_permutation_importance.png")


# ---------------------------------------------------------------------------
# 08 -- Random Forest depth tuning, VALIDATION set only (notebook 4.5)
# ---------------------------------------------------------------------------
def depth_tuning():
    labels = ["None\n(unlimited)", "4", "6", "8", "12"]
    values = [54.5, 61.6, 61.8, 61.8, 61.3]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(values))

    # guides that turn the overfit into a visible vertical distance
    for level in (54.5, 61.8):
        ax.axhline(level, color=GRID, linewidth=1.0, linestyle="--", zorder=1)

    ax.plot(x, values, color=BLUE, linewidth=2.4, marker="o", markersize=7,
            markerfacecolor="white", markeredgewidth=2, zorder=3)

    # the chosen setting
    ax.plot(3, values[3], marker="o", markersize=9, color=NAVY, zorder=4)
    ax.annotate("chosen: max_depth = 8", xy=(3, values[3]), xytext=(3, 58.3),
                ha="center", fontsize=9.5, color=NAVY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVY, linewidth=1.2))

    # value labels: below the outlier point, above the plateau
    for xi, v in zip(x, values):
        below = xi == 0
        ax.text(xi, v - 0.55 if below else v + 0.5, f"{v:.1f}%", ha="center",
                va="top" if below else "bottom", fontsize=9.5, color=INK)

    # make the overfitting gap the visual point
    ax.annotate("", xy=(0.62, 54.5), xytext=(0.62, 61.8),
                arrowprops=dict(arrowstyle="<->", color=RED, linewidth=1.4))
    ax.text(0.74, 58.15, "7.3-point\noverfitting gap", fontsize=9.5, color=RED,
            va="center", linespacing=1.3)

    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_xlabel("Random Forest max_depth")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(52, 64)
    ax.set_xlim(-0.35, 4.35)
    ax.grid(axis="x", visible=False)
    ax.set_title("An unlimited-depth forest memorized noise —\n"
                 "capping depth recovered 7 points", loc="left")
    note(fig, "Validation-set accuracy only. Depth was chosen on the "
              "validation set; the test set was reserved for the final model.",
         dy=-0.16)
    save(fig, "08_depth_tuning.png")


# ---------------------------------------------------------------------------
# 09 -- confusion matrix, Path A (notebook 04, Phase 4 80/20 split)
# ---------------------------------------------------------------------------
def confusion_matrix():
    cm = np.array([[5014, 3271, 19],
                   [2013, 6490, 396],
                   [116, 1905, 776]])
    classes = ["Low", "Medium", "High"]
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    # sequential blues -- counts have no polarity, so no diverging map here
    blues = LinearSegmentedColormap.from_list("blues", ["#ffffff", BLUE, NAVY])

    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    im = ax.imshow(cm, cmap=blues, vmin=0, vmax=cm.max())
    ax.grid(False)

    for i in range(3):
        for j in range(3):
            dark = cm[i, j] > cm.max() * 0.55
            ax.text(j, i - 0.09, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if dark else INK)
            ax.text(j, i + 0.19, f"{row_pct[i, j]:.0f}% of row", ha="center",
                    va="center", fontsize=9,
                    color="white" if dark else INK_SOFT)

    ax.set_xticks(range(3), [f"predicted\n{c}" for c in classes], fontsize=10)
    ax.set_yticks(range(3), [f"true {c}" for c in classes], fontsize=10)
    ax.set_xlabel("")
    fig.colorbar(im, ax=ax, shrink=0.78, label="people")
    ax.set_title("Mistakes are near-misses into Medium —\n"
                 "Low is almost never called High", loc="left")
    note(fig, "Path A (Linear Regression → buckets), notebook 04 / Phase 4, "
              "80/20 stratified split, n = 20,000. Overall accuracy 0.614.")
    save(fig, "09_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 10 -- largest extension-model coefficients (notebook 4.5)
# ---------------------------------------------------------------------------
def extension_effects():
    effects = [
        ("Shift work", -1.004),
        ("Mental health: healthy", 0.887),
        ("Weekend", 0.662),
        ("Mental health: anxiety + depression", -0.590),
        ("Occupation: freelancer", 0.575),
        ("Occupation: retired", 0.527),
        ("Occupation: homemaker", 0.523),
        ("Occupation: teacher", 0.420),
        ("Stress (per point, of 9)", -0.406),
        ("Morning chronotype", 0.321),
    ]
    effects.sort(key=lambda kv: abs(kv[1]))       # largest at top of barh
    labels = [k for k, _ in effects]
    values = [v for _, v in effects]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.arange(len(values))
    colors = [RED if v < 0 else BLUE for v in values]
    bars = ax.barh(y, values, color=colors, height=0.66,
                   edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in values], padding=4,
                 fontsize=9.5, color=INK)

    ax.axvline(0, color=INK_SOFT, linewidth=1.1)
    ax.set_yticks(y, labels, fontsize=9.5)
    ax.set_xlabel("Effect on sleep quality (points on the 1–10 scale),\n"
                  "holding every other variable constant")
    ax.set_xlim(-1.35, 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Shift work costs a full point of sleep quality", loc="left")
    note(fig, "Stress looks small here because it is measured per point: "
              "−0.406 across the 9-point stress range is a ~3.7-point total "
              "swing, larger than any single effect shown.")
    save(fig, "10_extension_effects.png")


# ---------------------------------------------------------------------------
# 11 -- fairness audit by occupation, Path A (notebook 04, Phase 4 test set)
# ---------------------------------------------------------------------------
def fairness_by_occupation():
    rows = [
        ("Lawyer", 0.711, 1003), ("Nurse", 0.694, 2003),
        ("Doctor", 0.681, 1563), ("Driver", 0.637, 1397),
        ("Retired", 0.614, 1438), ("Student", 0.601, 2971),
        ("Sales", 0.591, 1380), ("Homemaker", 0.590, 1199),
        ("Manager", 0.580, 1587), ("Freelancer", 0.579, 1453),
        ("Software Engineer", 0.571, 2443), ("Teacher", 0.561, 1563),
    ]
    overall = 0.614
    rows = sorted(rows, key=lambda r: r[1])        # best ends up at the top
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    ns = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    y = np.arange(len(values))
    colors = [BLUE if v >= overall else RED for v in values]
    bars = ax.barh(y, values, color=colors, height=0.68,
                   edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{v:.3f}   n={n:,}" for v, n in zip(values, ns)],
                 padding=4, fontsize=9, color=INK)

    ax.axvline(overall, color=INK_SOFT, linestyle="--", linewidth=1.2,
               zorder=3)
    ax.text(overall + 0.008, len(values) - 0.35, "overall 61.4%", fontsize=9,
            color=INK_SOFT, va="center")
    ax.set_ylim(-0.7, len(values) - 0.05)

    ax.set_yticks(y, labels, fontsize=9.5)
    ax.set_xlabel("Path A accuracy on the held-out test set")
    ax.set_xlim(0, 0.92)
    ax.grid(axis="y", visible=False)
    ax.set_title("The headline number hides a 15.0-point spread\n"
                 "between occupations", loc="left")
    note(fig, "Path A on the Phase 4 test set (n = 20,000). Blue is at or "
              "above the overall accuracy, red below it. Lawyer 0.711 down "
              "to Teacher 0.561.")
    save(fig, "11_fairness_by_occupation.png")


# ---------------------------------------------------------------------------
# 12 -- per-class recall, primary Path A vs extension Gradient Boosting
# ---------------------------------------------------------------------------
def class_recall():
    classes = ["Low", "Medium", "High"]
    primary = [0.60, 0.73, 0.28]
    extension = [0.73, 0.73, 0.44]

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    x = np.arange(len(classes))
    w = 0.36
    b1 = ax.bar(x - w / 2, primary, w, label="Primary — Path A",
                color=BLUE, edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w / 2, extension, w,
                label="Extension — Gradient Boosting",
                color=NAVY, edgecolor="white", linewidth=0.8)
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=10, color=INK)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=10, color=INK)

    ax.set_xticks(x, classes, fontsize=11)
    ax.set_xlabel("True sleep-quality class")
    ax.set_ylabel("Recall (share of that class found)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9.5)
    ax.set_title("The extension helps most where the model was worst —\n"
                 "but 'High' stays the weakest class for both", loc="left")
    note(fig, "Primary Path A: notebook 04 / Phase 4 test set, n = 20,000. "
              "Extension Gradient Boosting: notebook 4.5 / Phase 4.5 test "
              "set, n = 15,000.")
    save(fig, "12_class_recall.png")


if __name__ == "__main__":
    model_scorecard()
    permutation_importance()
    depth_tuning()
    confusion_matrix()
    extension_effects()
    fairness_by_occupation()
    class_recall()
