
import numpy as np
import pandas as pd
from data_module.types import AnnotatedFeaturesCollection
def get_tsne_dataframe(
        data: np.ndarray,
        labels: list[str],
        partitions: list[str] | None
    ) -> pd.DataFrame:
    """Creates a DataFrame from t-SNE results.
    
    Args:
        data: Holds the component-1, component-2 from t-SNE
        labels: Holds the labels for each data point
        partitions: Holds the partition marker for each data point (e.g. square, circle, pentagon)
            for extra reference https://plotly.com/python/marker-style/
    """
    df = pd.DataFrame(data, columns=['component-1', 'component-2'])
    df['label'] = labels
    if partitions is not None:
        df['shape'] = partitions
    return df
