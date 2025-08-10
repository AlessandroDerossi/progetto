
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    symbol: str | None = None,
) -> go.Figure:
    """
    Create a scatter plot using Plotly.

    Args:
        df: Pandas dataframe holding the data to plot
        x_col: Name of the column to be plotted on the x-axis
        y_col: Name of the column to be plotted on the y-axis
        color_col: Name of the column to be used for coloring the points
        symbol: Name of the column to be used for symbolizing the points (optional)
            Can be useful to differentiate between train/test splits.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        symbol=symbol,
        title=title,
    )
    fig.update_layout(template="plotly_white")

    return fig