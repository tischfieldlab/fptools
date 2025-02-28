from typing import List, Tuple, Union
import seaborn as sns
import matplotlib as mpl

Palette = Union[str, List[Union[str, Tuple[float, float, float]]], None]


def get_colormap(palette: Palette) -> mpl.colors.Colormap:
    """Generate a matplotlib Colormap from a `Palette`.

    Args:
        palette: if None, a diverging palette will be created. If a str, will fetch the corresponding matplotlib colormap. If a list, will generate a matplotlib listed colormap.

    Returns:
        a matplotlib colormap instance.
    """
    if palette is None:
        return sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    elif isinstance(palette, str):
        return mpl.colormaps[palette]
    elif isinstance(palette, list):
        return mpl.colors.ListedColormap(palette)
