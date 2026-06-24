"""Figures and the evaluation writeup."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_index_map(da, path, title="NDCI"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(da.values, cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_map(raster, path, title="Embedding anomaly (cosine distance)"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(raster, cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(anom_df, rephy_df, path):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(anom_df["date"], anom_df["distance"], "o-", color="crimson", label="embedding anomaly")
    ax1.set_ylabel("embedding anomaly", color="crimson")
    ax1.set_xlabel("2018")
    if rephy_df is not None and len(rephy_df):
        ax2 = ax1.twinx()
        for p, g in rephy_df.groupby("param"):
            ax2.plot(g["date"], g["value"], "s--", alpha=0.6, label=p)
        ax2.set_ylabel("REPHY in-situ")
        ax2.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_evaluation(metrics, path):
    lines = ["# Evaluation\n"]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
