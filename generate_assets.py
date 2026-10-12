"""
generate_assets.py
Regenerates every README visual into ./assets.

Run with the scikit-learn version the model was trained with (1.6.1):
    pip install "scikit-learn==1.6.1" matplotlib pandas joblib
    python generate_assets.py --model "RF Mental Health Model.pkl"
"""
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

# ---------- theme ----------
BG, PANEL, INK, MUTED = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
CYAN, PURPLE, GREEN, ORANGE, PINK, YELLOW = (
    "#58a6ff", "#bc8cff", "#3fb950", "#f0883e", "#ff7b72", "#e3b341")
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": "#30363d", "font.family": "DejaVu Sans",
})

# ---------- real numbers copied from the notebook output ----------
RESULTS = pd.DataFrame({
    "Model": ["Linear\nRegression", "Random Forest\n(default, shipped)", "Random Forest\n(tuned)"],
    "R2_test": [0.739794, 0.877589, 0.865014],
    "R2_train": [0.723677, 0.980829, 0.954703],
    "MAE": [0.536178, 0.347221, 0.368902],
    "RMSE": [0.676032, 0.463681, 0.486915],
})


# =====================================================================
# 1. 2D architecture
# =====================================================================
def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=PANEL, ec=color, lw=2.2))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=8.5, color=MUTED, linespacing=1.4)


def arrow(ax, p1, p2, color=MUTED, label=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,
                                 color=color, lw=2))
    if label:
        ax.text(p1[0] - 0.15, (p1[1] + p2[1]) / 2, label,
                ha="right", va="center", fontsize=8.5, color=color)


def architecture_2d():
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(7.5, 6.85, "System Architecture", ha="center", fontsize=20, fontweight="bold")
    ax.text(7.5, 6.45, "Offline training  →  serialized pipeline  →  online inference",
            ha="center", fontsize=10.5, color=MUTED)

    # offline lane
    ax.text(0.3, 5.75, "OFFLINE  ·  Jupyter", fontsize=9.5, color=YELLOW, fontweight="bold")
    box(ax, 0.3, 4.15, 2.5, 1.35, "Raw CSV", "5,000 students\n13 columns", YELLOW)
    box(ax, 3.4, 4.15, 2.7, 1.35, "Clean + Engineer", "dedupe · clip(≥0)\ntop-10 country group", ORANGE)
    box(ax, 6.7, 4.15, 3.0, 1.35, "ColumnTransformer", "log1p+scale · scale\nordinal · one-hot", PURPLE)
    box(ax, 10.3, 4.15, 2.5, 1.35, "RandomForest", "Pipeline.fit()\n70/30 split", GREEN)
    box(ax, 13.1, 4.15, 1.7, 1.35, ".pkl", "joblib.dump\n~25 MB", PINK)
    for a, b in [(2.8, 3.4), (6.1, 6.7), (9.7, 10.3), (12.8, 13.1)]:
        arrow(ax, (a + 0.02, 4.82), (b - 0.02, 4.82), YELLOW)
    arrow(ax, (13.95, 4.1), (13.95, 3.05), PINK, "load once at startup")

    # online lane
    ax.text(0.3, 2.95, "ONLINE  ·  FastAPI", fontsize=9.5, color=CYAN, fontweight="bold")
    box(ax, 0.3, 0.9, 2.5, 1.5, "Client", "Swagger UI · curl\nany frontend (CORS *)", CYAN)
    box(ax, 3.4, 0.9, 2.7, 1.5, "POST /predict", "JSON body", CYAN)
    box(ax, 6.7, 0.9, 3.0, 1.5, "Pydantic Validation", "ranges · Literal enums\n422 on bad input", PURPLE)
    box(ax, 10.3, 0.9, 2.5, 1.5, "Feature Row", "country → top-10\nor 'Other'", ORANGE)
    box(ax, 13.1, 0.9, 1.7, 1.5, "predict()", "round(…, 2)", GREEN)
    for a, b in [(2.8, 3.4), (6.1, 6.7), (9.7, 10.3), (12.8, 13.1)]:
        arrow(ax, (a + 0.02, 1.65), (b - 0.02, 1.65), CYAN)
    ax.text(7.5, 0.25, "Response:  { \"predicted_mental_health_score\": 7.42 }",
            ha="center", fontsize=11, family="monospace", color=GREEN)
    fig.savefig(ASSETS / "architecture_2d.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 2. 3D isometric layered architecture
# =====================================================================
def slab(ax, x, y, z, w, d, h, color, alpha=0.92):
    v = np.array([[x, y, z], [x + w, y, z], [x + w, y + d, z], [x, y + d, z],
                  [x, y, z + h], [x + w, y, z + h], [x + w, y + d, z + h], [x, y + d, z + h]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    shades = [0.55, 1.0, 0.75, 0.75, 0.65, 0.65]
    rgb = np.array(matplotlib_rgb(color))
    polys = [v[f] for f in faces]
    cols = [tuple(np.clip(rgb * s, 0, 1)) + (alpha,) for s in shades]
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=BG, linewidths=1.2)
    ax.add_collection3d(pc)


def matplotlib_rgb(hexcolor):
    h = hexcolor.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def architecture_3d():
    from mpl_toolkits.mplot3d import proj3d
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes([0.0, 0.0, 0.58, 0.95], projection="3d")
    ax.set_facecolor(BG)
    layers = [  # (label, sublabel, color)  bottom -> top
        ("1 · DATA", "Student_Social_Media_And_Mental_Health_Impact.csv", YELLOW),
        ("2 · PREPROCESSING", "ColumnTransformer: log1p, scaler, ordinal, one-hot", ORANGE),
        ("3 · MODEL", "RandomForestRegressor on 38 encoded features", GREEN),
        ("4 · PIPELINE ARTIFACT", "joblib .pkl: preprocessing + model in one object", PINK),
        ("5 · API", "FastAPI: Pydantic validation, CORS, /predict", PURPLE),
        ("6 · CLIENT", "Swagger UI, browser, curl", CYAN),
    ]
    gap, h = 1.6, 0.55
    for i, (_, _, col) in enumerate(layers):
        slab(ax, 0, 0, i * gap, 6, 4, h, col)
    ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.set_zlim(0, len(layers) * gap)
    ax.set_box_aspect((1.5, 1, 1.9))
    ax.view_init(elev=20, azim=-62)
    ax.set_axis_off()
    fig.canvas.draw()

    # project right-hand corner of each slab to figure coords, draw leader lines + labels
    for i, (name, sub, col) in enumerate(layers):
        x2, y2, _ = proj3d.proj_transform(6, 4, i * gap + h / 2, ax.get_proj())
        disp = ax.transData.transform((x2, y2))
        fx, fy = fig.transFigure.inverted().transform(disp)
        ty = 0.12 + i * 0.145
        fig.add_artist(plt.Line2D([fx, 0.60, 0.63], [fy, ty, ty], color=col, lw=1.6,
                                  transform=fig.transFigure))
        fig.text(0.635, ty + 0.012, name, color=col, fontsize=13, fontweight="bold", va="bottom")
        fig.text(0.635, ty - 0.008, sub, color=MUTED, fontsize=9, va="top")
    fig.suptitle("Layered Stack  \u00b7  3D View", fontsize=20, fontweight="bold", x=0.5, y=0.98)
    fig.savefig(ASSETS / "architecture_3d.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 3. Model comparison + overfitting gap
# =====================================================================
def comparison_charts():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = [MUTED, GREEN, CYAN]
    x = np.arange(3)

    axes[0].bar(x, RESULTS.R2_test, color=colors, width=0.62)
    axes[0].set_title("R²  (test, higher is better)", fontweight="bold")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(RESULTS.R2_test):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    axes[1].bar(x - 0.2, RESULTS.MAE, width=0.38, color=ORANGE, label="MAE")
    axes[1].bar(x + 0.2, RESULTS.RMSE, width=0.38, color=PINK, label="RMSE")
    axes[1].set_title("Error in score points  (lower is better)", fontweight="bold")
    axes[1].legend(facecolor=PANEL, edgecolor="#30363d")
    for i in range(3):
        axes[1].text(i - 0.2, RESULTS.MAE[i] + 0.012, f"{RESULTS.MAE[i]:.2f}", ha="center", fontsize=9)
        axes[1].text(i + 0.2, RESULTS.RMSE[i] + 0.012, f"{RESULTS.RMSE[i]:.2f}", ha="center", fontsize=9)

    axes[2].bar(x - 0.2, RESULTS.R2_train, width=0.38, color=PURPLE, label="Train R²")
    axes[2].bar(x + 0.2, RESULTS.R2_test, width=0.38, color=CYAN, label="Test R²")
    axes[2].set_title("Generalisation gap", fontweight="bold")
    axes[2].set_ylim(0, 1.12)
    axes[2].legend(facecolor=PANEL, edgecolor="#30363d", loc="lower right")
    for i in range(3):
        gap = RESULTS.R2_train[i] - RESULTS.R2_test[i]
        axes[2].text(i, 1.03, f"gap {gap:+.3f}", ha="center", fontsize=9, color=YELLOW)

    for a in axes:
        a.set_xticks(x); a.set_xticklabels(RESULTS.Model, fontsize=8.5)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#30363d", lw=0.6); a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS / "model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 4. Feature importance (grouped back to original columns)  +  3D surface
# =====================================================================
def load_model(path):
    model = joblib.load(path)
    pre, rf = model.steps[0][1], model.steps[-1][1]
    return model, pre, rf


def feature_importance(pre, rf):
    nom_cols = list(pre.transformers_[3][2])
    ohe = pre.transformers_[3][1].steps[-1][1]
    names = (["Study_Hours", "Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks",
              "Physical_Activity_Hours", "Sleep_Hours_Per_Night", "Stress_Level"]
             + list(ohe.get_feature_names_out()))
    imp = pd.Series(rf.feature_importances_, index=names)
    grouped = {}
    for n, v in imp.items():
        key = next((c for c in nom_cols if n.startswith(c + "_")), n)
        key = "Country (grouped)" if key.startswith("Country_Grouped") or key.startswith("Grouped_country") else key
        grouped[key] = grouped.get(key, 0) + v
    s = pd.Series(grouped).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = cm.cool(np.linspace(0.15, 0.95, len(s)))
    ax.barh(s.index, s.values, color=palette, height=0.7)
    for i, v in enumerate(s.values):
        ax.text(v + 0.008, i, f"{v * 100:.1f}%", va="center", fontsize=9)
    ax.set_title("Random Forest feature importance  (one-hot columns re-grouped)",
                 fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#30363d", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS / "feature_importance.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return s


def base_row(pre):
    country_col = "Country_Grouped" if "Country_Grouped" in pre.feature_names_in_ else "Grouped_country"
    return dict(Age=21, Gender="Male", Academic_Level="Undergraduate",
                Most_Used_Platform="Instagram", Purpose_Of_Use="Entertainment",
                Avg_Daily_Usage_Hours=5.0, Daily_Unlocks=170, Study_Hours=3.0,
                Physical_Activity_Hours=1.5, Sleep_Hours_Per_Night=7.0,
                Stress_Level="Medium", **{country_col: "India"})


def surface_3d(model, pre):
    base = base_row(pre)
    usage = np.linspace(1, 9, 28)
    sleep = np.linspace(4, 9, 28)
    U, S = np.meshgrid(usage, sleep)
    rows = []
    for u, s in zip(U.ravel(), S.ravel()):
        r = dict(base); r["Avg_Daily_Usage_Hours"] = u; r["Sleep_Hours_Per_Night"] = s
        rows.append(r)
    Z = model.predict(pd.DataFrame(rows)).reshape(U.shape)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(U, S, Z, cmap="plasma", edgecolor="none", alpha=0.95, rstride=1, cstride=1)
    ax.contourf(U, S, Z, zdir="z", offset=Z.min() - 0.6, cmap="plasma", alpha=0.35)
    ax.set_xlabel("Daily social-media hours", labelpad=10)
    ax.set_ylabel("Sleep hours / night", labelpad=10)
    ax.set_zlabel("Predicted score", labelpad=8)
    ax.set_zlim(Z.min() - 0.6, Z.max() + 0.3)
    ax.view_init(elev=26, azim=-132)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.09, 0.11, 0.14, 1.0))
    fig.colorbar(surf, shrink=0.55, pad=0.08, label="Predicted Mental_Health_Score")
    ax.set_title("What the shipped model has learned\n(other inputs held at a fixed profile)",
                 fontsize=15, fontweight="bold", pad=18)
    fig.savefig(ASSETS / "prediction_surface_3d.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 2D companion: stress ladder across usage
    fig, ax = plt.subplots(figsize=(9, 5))
    for stress, col in zip(["Low", "Medium", "High", "Very High"], [GREEN, CYAN, ORANGE, PINK]):
        ys = []
        for u in usage:
            r = dict(base); r["Avg_Daily_Usage_Hours"] = u; r["Stress_Level"] = stress
            ys.append(float(model.predict(pd.DataFrame([r]))[0]))
        ax.plot(usage, ys, color=col, lw=2.6, label=f"Stress: {stress}")
    ax.set_xlabel("Daily social-media hours"); ax.set_ylabel("Predicted score")
    ax.set_title("Predicted score vs. usage, by stress level", fontweight="bold", loc="left")
    ax.legend(facecolor=PANEL, edgecolor="#30363d")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#30363d", lw=0.6)
    fig.tight_layout()
    fig.savefig(ASSETS / "stress_ladder.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RF Mental Health Model.pkl")
    args = ap.parse_args()
    architecture_2d()
    architecture_3d()
    comparison_charts()
    mdl, pre, rf = load_model(args.model)
    print(feature_importance(pre, rf).round(4).to_string())
    surface_3d(mdl, pre)
    print("assets written to", ASSETS.resolve())
