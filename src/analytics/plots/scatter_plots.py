import typing as t

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from analytics import DataFinal
from analytics.definitions import Column, Label, SOTA, OrderedCategory
from analytics.plots.plot_utils import network_size_in_log_scale, figure, merge_network_size_with, \
    mark_unknown
from analytics.data.transforms import ModelsTF, ApplicationsTF, ArchitecturesTF, LearningTF, BenchmarkTF, \
    apply_transform


def _order_result_categories_inplace(df: pd.DataFrame) -> None:
    """Inplace ordering of models and architectures

    This permits hiding unreferenced classes in the plot legend, hence cannot be applied beforehand
    """
    for (column, labels) in [
        (BenchmarkTF.ModelCol, OrderedCategory.Models),
        (BenchmarkTF.ArchitectureCol, OrderedCategory.Architectures)
    ]:
        unique = df[column].unique()
        actual = [label for label in labels if label in unique]
        df[column] = pd.Categorical(values=df[column], categories=actual, ordered=True)


def _get_results(dfs: t.Dict[str, pd.DataFrame], benchmark: str, min_score: float = 0.0) -> pd.DataFrame:
    # Note manual_jitter injects gaussian noise into Year column to spread results over X axis (timeline).
    # We do so because `sb.stripplot` does not support variable marker style based on another column, hence adding noise
    # at this stage is the simplest lossless solution.
    df = apply_transform(dfs, DataFinal.BENCHMARK.name, BenchmarkTF, manual_jitter=True)
    df = df[df[Column.BENCHMARK] == benchmark]
    if min_score:
        df = df[df['Result'] >= min_score]
    _order_result_categories_inplace(df)
    return df


def _scatter_benchmark_results(df: pd.DataFrame, min_score: float = 0.0) -> None:
    sb.scatterplot(x=Column.YEAR, y='Result', hue='Neuron', style='Architecture', data=df, sizes=7)
    plt.ylim([min_score if min_score else df['Result'].min()-0.5, 100.])
    plt.xlim([df[Column.YEAR].min()-0.2, df[Column.YEAR].max()+0.2])
    plt.xlim([1990, 2022])
    plt.ylabel("Classification Accuracy (%)", weight=600, fontsize="large")
    plt.xlabel(Label.YEAR, weight=600, fontsize="large")


def _plot_sota_result(benchmark: str, hide_sota: bool, axes: plt.Axes, lims: t.Tuple[float, float]):
    if not hide_sota and benchmark in SOTA:
        bm = SOTA[benchmark]
        axes.plot(lims, [bm.sota, bm.sota], '--r', label=f"{bm.holder} ({bm.sota})")
        # plt.legend()  # Breaks seaborn's indentation


@mark_unknown
def _scatter_network_size_over_time(df: pd.DataFrame, *, labels: t.List[str], column: str) -> None:
    df[column] = pd.Categorical(df[column], categories=labels, ordered=True)
    sb.stripplot(x=Column.YEAR, y=Column.NETWORK_SIZE, hue=column, data=df, size=6, marker="^", jitter=0.3)
    network_size_in_log_scale(df[Column.NETWORK_SIZE].max())
    handles, labels = plt.xticks()
    plt.xticks(handles[::5], labels[::5])
    plt.xlabel(Label.YEAR, weight=600, fontsize='large')
    plt.legend()


@figure(showable=True, saveable=True, tight=True)
def scatter_network_size_over_time_per_model(dfs: t.Dict[str, pd.DataFrame], group_if: bool = False, **kwargs) -> None:
    df = merge_network_size_with(dfs, DataFinal.MODELS.name, ModelsTF, group_if=group_if)
    labels = OrderedCategory.ModelsSingleIF if group_if else OrderedCategory.Models
    _scatter_network_size_over_time(df, labels=labels, column=Column.NEURON_MODEL, **kwargs)


@figure(showable=True, saveable=True, tight=True)
def scatter_network_size_over_time_per_application(
        dfs: t.Dict[str, pd.DataFrame], practical_only: bool = False, **kwargs
) -> None:
    df = merge_network_size_with(dfs, DataFinal.APPLICATIONS.name, ApplicationsTF, practical_only=practical_only)
    labels = OrderedCategory.ApplicationsPractical if practical_only else OrderedCategory.Applications
    _scatter_network_size_over_time(df, labels=labels, column=Column.APPLICATION, **kwargs)


@figure(showable=True, saveable=True, tight=True)
def scatter_network_size_over_time_per_architecture(dfs: t.Dict[str, pd.DataFrame], **kwargs) -> None:
    df = merge_network_size_with(dfs, DataFinal.ARCHITECTURE.name, ArchitecturesTF)
    _scatter_network_size_over_time(df, labels=OrderedCategory.Architectures, column=Column.ARCHITECTURE, **kwargs)


@figure(showable=True, saveable=True, tight=True)
def scatter_network_size_over_time_per_learning(dfs: t.Dict[str, pd.DataFrame], **kwargs) -> None:
    df = merge_network_size_with(dfs, DataFinal.LEARNING.name, LearningTF)
    _scatter_network_size_over_time(df, labels=OrderedCategory.LearningAlgorithms, column=Column.LEARNING, **kwargs)


@figure(showable=True, saveable=True, tight=True)
def scatter_network_size_summary(dfs: t.Dict[str, pd.DataFrame]) -> None:
    gb = dfs[DataFinal.NETWORKS.name].drop(columns=[Column.PUBLICATION]).groupby(Column.YEAR)
    df = gb.aggregate(['max', 'median', 'min']).droplevel(0, axis=1).reset_index()
    # Add trend on median data using 2nd order polynomial fit
    sb.regplot(x=Column.YEAR, y='median', data=df, ci=0, scatter=False, label='Trend (median)')
    sb.scatterplot(x=Column.YEAR, y='median', data=df, marker='o', label='Median')
    sb.scatterplot(x=Column.YEAR, y='max', data=df, marker='s', label='Max')
    sb.scatterplot(x=Column.YEAR, y='min', data=df, marker='v', label='Min')
    network_size_in_log_scale(df['max'].max())
    plt.xlabel(Label.YEAR, weight=600, fontsize='large')
    plt.legend()


@figure(showable=True, saveable=True, tight=True)
def scatter_benchmark_results(
        dfs: t.Dict[str, pd.DataFrame],
        *,
        benchmark: str,
        min_score: float = 0.0,
        hide_sota: bool = False
) -> None:
    # Note manual_jitter injects gaussian noise into Year column to spread results over X axis (timeline).
    # We do this because sb.stripplot does not support variable marker style based on another column, so this is the
    # simplest lossless solution.
    df = _get_results(dfs, benchmark, min_score)
    _scatter_benchmark_results(df, min_score)
    _plot_sota_result(benchmark, hide_sota, plt, plt.xlim())


@figure(showable=True, saveable=True, tight=True)
def scatter_benchmark_results_closeup(
        dfs: t.Dict[str, pd.DataFrame],
        *,
        benchmark: str,
        min_score: float = 0.0,
        hide_sota: bool = False
) -> None:
    df = _get_results(dfs, benchmark)

    with plt.rc_context({'legend.loc': 'lower left'}):
        _scatter_benchmark_results(df)
        _plot_sota_result(benchmark, hide_sota, plt, plt.xlim())

    fig = plt.gcf()
    ax = fig.add_axes(rect=[0.6, 0.2, 0.35, 0.35])

    # Note we do not filter by result before plotting to retain the same color/marker assignment
    sb.scatterplot(x=Column.YEAR, y='Result', hue='Neuron', style='Architecture', data=df, sizes=7, axes=ax)
    ds = df[df['Result'] >= min_score][Column.YEAR]
    ax.set_xlim([ds.min(), ds.max()])
    ax.set_ylim([min_score, 100.])
    _plot_sota_result(benchmark, hide_sota, ax, ax.get_xlim())

    ax.get_legend().remove()
    ax.set_ylabel('')
    ax.set_xlabel('')


@figure(showable=True, saveable=True, tight=True)
def scatter_benchmark_results_multiple(dfs: t.Dict[str, pd.DataFrame], what: t.List[str], ncols: int = 3) -> None:
    fig: plt.Figure = plt.gcf()
    nrows = len(what) // ncols
    fig.subplots(nrows, ncols)

    def to_location(i):
        return int(f'{nrows:.0f}{ncols:.0f}{i+1:.0f}')

    with plt.rc_context({'legend.loc': 'lower left'}):
        for idx, benchmark in enumerate(what):
            scatter_benchmark_results(dfs, benchmark=benchmark, fid=fig.number, subplot=to_location(idx), min_score=60)
            ax: plt.Axes = plt.gca()
            ax.set_title(f' {benchmark}', loc='left', y=1.0, pad=-15)
