import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensor import ContradictionTensor


def heatmap(
    ct: ContradictionTensor,
    context: str | None = None,
    output_path: str = "heatmap.png",
) -> None:
    if len(ct.beliefs) < 2:
        raise ValueError("need at least 2 beliefs to render a heatmap")

    ctx_idx = ct._ctx_idx(context)
    m = ct._tensor[:, :, ctx_idx].copy()
    labels = [b.id for b in ct.beliefs]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))

    sns.heatmap(
        m,
        mask=np.eye(n, dtype=bool),
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
        cbar_kws={"label": "contradiction score", "shrink": 0.8},
    )

    title = "Contradiction Tensor"
    if context:
        title += f"  [{context}]"
    ax.set_title(title, pad=14, fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
