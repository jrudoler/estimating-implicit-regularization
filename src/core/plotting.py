import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cmap import Colormap
from pandas import DataFrame


_HUE_CMAP = Colormap("crameri:imola").to_mpl()


def panelplot(data: DataFrame, x: str, y: str, hue: str = None, **kwargs) -> None:
    """Panel plot of x vs y, with hue. Suitable for use with seaborn.FacetGrid.

    Args:
        data: A pandas DataFrame containing the data to plot.
        x: The name of the column to plot on the x-axis.
        y: The name of the column to plot on the y-axis.
        hue: The name of the column to use for the hue.
        **kwargs: Additional arguments to pass to the plot.

    Example:
    ```python
    g = sns.FacetGrid(data, col="width", row="depth")
    g.map_dataframe(panelplot, x="dropout", y="estimated_bias", hue="train_bias/loss")
    ```
    """
    ax = plt.gca()
    ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5)
    if hue is not None:
        norm = mpl.colors.Normalize(vmin=data[hue].min(), vmax=data[hue].max())
        palette = _HUE_CMAP
    else:
        norm = None
        palette = None
    # plot violin plot
    sns.violinplot(
        data=data,
        x=x,
        y=y,
        inner=None,
        cut=0,
        density_norm="width",
        color="lightgray",
        alpha=0.5,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        size=5,
        linewidth=0.5,
        hue=hue,
        hue_norm=norm,
        alpha=0.6,
        palette=palette,
        edgecolor="auto",
        ax=ax,
    )
