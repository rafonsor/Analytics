import typing as t

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from analytics import DataFinal
from analytics.definitions import Column, Label, OrderedCategory
from analytics.plots.plot_utils import network_size_in_log_scale, figure, merge_network_size_with, \
    mark_unknown
from analytics.data.transforms import ModelsTF, ApplicationsTF, ArchitecturesTF, LearningTF


@mark_unknown
def _network_size_box_plots(df: pd.DataFrame, *, labels: t.List[str], column: str, xlabel: str) -> None:
    # Impose categorical ordering in Seaborn plots
    df[column] = pd.Categorical(df[column], categories=labels, ordered=True)
    sb.boxplot(x=column, y=Column.NETWORK_SIZE, data=df, whis=np.inf, boxprops={"alpha": 0.2})
    sb.stripplot(x=column, y=Column.NETWORK_SIZE, data=df, size=4, marker='^')
    network_size_in_log_scale(df[Column.NETWORK_SIZE].max())
    plt.xticks(rotation=40)
    plt.xlabel(xlabel, weight=600, fontsize='large')


@figure(showable=True, saveable=True, tight=True)
def boxplot_network_size_per_model(dfs: t.Dict[str, pd.DataFrame]) -> None:
    df = merge_network_size_with(dfs, DataFinal.MODELS.name, ModelsTF)
    _network_size_box_plots(df, labels=OrderedCategory.Models, column=Column.NEURON_MODEL, xlabel=Label.NEURON_MODEL)


@figure(showable=True, saveable=True, tight=True)
def boxplot_network_size_per_application(dfs: t.Dict[str, pd.DataFrame]) -> None:
    df = merge_network_size_with(dfs, DataFinal.APPLICATIONS.name, ApplicationsTF)
    _network_size_box_plots(
        df, labels=OrderedCategory.Applications, column=Column.APPLICATION, xlabel=Label.APPLICATION)


@figure(showable=True, saveable=True, tight=True)
def boxplot_network_size_per_architecture(dfs: t.Dict[str, pd.DataFrame]) -> None:
    df = merge_network_size_with(dfs, DataFinal.ARCHITECTURE.name, ArchitecturesTF)
    _network_size_box_plots(
        df, labels=OrderedCategory.Architectures, column=Column.ARCHITECTURE, xlabel=Label.ARCHITECTURE)


@figure(showable=True, saveable=True, tight=True)
def boxplot_network_size_per_learning(dfs: t.Dict[str, pd.DataFrame]) -> None:
    df = merge_network_size_with(dfs, DataFinal.LEARNING.name, LearningTF)
    _network_size_box_plots(
        df, labels=OrderedCategory.LearningAlgorithms, column=Column.LEARNING, xlabel=Label.LEARNING)
