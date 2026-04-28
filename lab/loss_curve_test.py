import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def overlap_loss_from_pairwise_overlap_area(pairwise_overlap_area: torch.Tensor) -> torch.Tensor:
    """Apply the same masking and normalization used in placement.py."""
    n = pairwise_overlap_area.shape[0]
    mask = torch.triu(torch.ones_like(pairwise_overlap_area), diagonal=1)
    overlap_sclar = 150
    # normalization = torch.sqrt(
    #     torch.tensor(
    #         n,
    #         device=pairwise_overlap_area.device,
    #         dtype=pairwise_overlap_area.dtype,
    #     )
    # )
    # return torch.log1p(torch.sum(pairwise_overlap_area * mask) / normalization)
    return torch.log1p(torch.sum(pairwise_overlap_area * mask)) * overlap_sclar


def main() -> None:
    # Sweep one active upper-triangular overlap value from 1e7 down to 1.
    overlap_values = torch.logspace(7, 0, steps=200)
    losses = []

    for overlap_value in overlap_values:
        pairwise_overlap_area = torch.tensor(
            [
                [0.0, overlap_value.item()],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        losses.append(overlap_loss_from_pairwise_overlap_area(pairwise_overlap_area).item())

    losses = torch.tensor(losses)
    overlap_np = overlap_values.numpy()
    losses_np = losses.numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    (line,) = ax.plot(
        overlap_np,
        losses_np,
        linewidth=2,
        label="loss(pairwise_overlap_area)",
    )
    ax.scatter(overlap_np, losses_np, s=12, alpha=0.35, color=line.get_color())
    ax.set_xlabel("pairwise_overlap_area")
    ax.set_ylabel("loss")
    ax.set_title("Interactive Overlap Loss Curve")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "alpha": 0.6},
    )
    annotation.set_visible(False)

    def update_annotation(index: int) -> None:
        x_value = float(overlap_np[index])
        y_value = float(losses_np[index])
        annotation.xy = (x_value, y_value)
        annotation.set_text(
            f"pairwise_overlap_area={x_value:,.3f}\nloss={y_value:,.3f}"
        )
        annotation.set_visible(True)

    def on_move(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        display_points = ax.transData.transform(
            torch.tensor(list(zip(overlap_np, losses_np))).numpy()
        )
        dx = display_points[:, 0] - event.x
        dy = display_points[:, 1] - event.y
        distances = dx * dx + dy * dy
        nearest_index = int(distances.argmin())

        if distances[nearest_index] < 250:
            update_annotation(nearest_index)
        elif annotation.get_visible():
            annotation.set_visible(False)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
