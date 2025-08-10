

from data_module.types import AnnotatedFeaturesCollection
from ml.feature_extractor import compute_tsne
from plotting.dataframe import get_tsne_dataframe
from plotting.render import scatter_plot


def get_plot_tsne(
    data: AnnotatedFeaturesCollection,
    do_pca: bool = True,
    show: bool = False,
):
    tsne_results = compute_tsne(data, do_pca=do_pca)
    df = get_tsne_dataframe(
        data=tsne_results,
        labels=data.labels_as_str,
        partitions=None  # Assuming no partitioning for simplicity
    )

    figure = scatter_plot(
        df,
        x_col='component-1',
        y_col='component-2',
        color_col='label',
        symbol='shape' if 'shape' in df.columns else None,
        title="t-SNE Plot of Annotated Features",
    )
    if show:
        figure.show()