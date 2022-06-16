import typing as t

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from analytics import DataFinal
from analytics.definitions import Column, OrderedCategory
from analytics.plots.plot_utils import figure
from analytics.data.transforms import ArchitecturesTF, ApplicationsTF, ModelsTF, LearningTF

__ALL__ = [
    # 'grid_model_application',
    # 'grid_application_model',
]

DIM_A = 'a'
DIM_B = 'b'


def _grid(
        *,
        a: pd.DataFrame,
        b: pd.DataFrame,
        col_a: str,
        col_b: str,
        order_a: t.List[str],
        order_b: t.List[str],
        label_a: str,
        label_b: str
):
    # Join dataframes on overlapping references
    df = pd.merge(a, b, how='inner', on=Column.PUBLICATION).reset_index(drop=True)
    # Count publications per (a, b) value-pair
    df = df.groupby([col_a, col_b]).count().reset_index()
    # Restructure into 2D matrix of axb
    df = df.pivot_table(index=col_a, columns=col_b, values=Column.PUBLICATION)
    # Order a values
    df.index = pd.CategoricalIndex(df.index, categories=order_a, ordered=True)
    df.sort_index(inplace=True)
    # Order b values
    for missing in list(set(order_b) - set(df.columns)):
        df[missing] = np.NaN
    df = df[order_b]

    sb.heatmap(df, annot=True, linewidths=.5, cmap='rocket_r', vmin=0, fmt='.3g')
    plt.ylabel(label_a, weight=600, fontsize='large')
    plt.xlabel(label_b, weight=600, fontsize='large', rotation=0)


def _gen_grid_config(suffix, /, df, col, order, label) -> t.Dict[str, t.Any]:
    return {
        suffix: df,
        f'col_{suffix}': col,
        f'order_{suffix}': order,
        f'label_{suffix}': label,
    }


def grid_config_architecture(dfs, suffix) -> t.Dict[str, t.Any]:
    return _gen_grid_config(
        suffix,
        df=ArchitecturesTF.transform(dfs[DataFinal.ARCHITECTURE.name]),
        col=Column.ARCHITECTURE,
        order=OrderedCategory.Architectures,
        label='Network Architecture')


def grid_config_application(dfs, suffix, practical_only: bool = False) -> t.Dict[str, t.Any]:
    return _gen_grid_config(
        suffix,
        df=ApplicationsTF.transform(dfs[DataFinal.APPLICATIONS.name], practical_only=practical_only),
        col=Column.APPLICATION,
        order=OrderedCategory.ApplicationsPractical if practical_only else OrderedCategory.Applications,
        label='Application Area')


def grid_config_learning(dfs, suffix) -> t.Dict[str, t.Any]:
    return _gen_grid_config(
        suffix,
        df=LearningTF.transform(dfs[DataFinal.LEARNING.name]),
        col=Column.LEARNING,
        order=OrderedCategory.LearningAlgorithms,
        label='Learning Algorithm')


def grid_config_model(dfs, suffix, group_if: bool = False) -> t.Dict[str, t.Any]:
    return _gen_grid_config(
        suffix,
        df=ModelsTF.transform(dfs[DataFinal.MODELS.name], group_if=group_if),
        col=Column.NEURON_MODEL,
        order=OrderedCategory.ModelsSingleIF if group_if else OrderedCategory.Models,
        label='Model Class')


# from analytics import grid_plots
#
#
# def __gen_grid_func(x, y, dim_x, dim_y):
#     name = f'grid_{x}_{y}'
#
#     @showable
#     def grid_func(dfs):
#         print(name)
#         _grid(
#             **getattr(grid_plots, f'grid_config_{x}')(dfs, dim_x),
#             **getattr(grid_plots, f'grid_config_{y}')(dfs, dim_y),
#         )
#     grid_func.__name__ = name
#     setattr(grid_plots, name, grid_func)
#     __ALL__.append(name)
#
#
# for (first, second) in itertools.combinations(['architecture', 'learning', 'model'], 2):
#     __gen_grid_func(first, second, DIM_A, DIM_B)
#     __gen_grid_func(second, first, DIM_B, DIM_A)


@figure(showable=True, saveable=True, properties={"left": 0.22, "bottom": 0.34, "right": 1, "top": 0.98})
def grid_model_application(dfs, group_if: bool = False, practical_only: bool = False):
    _grid(
        **grid_config_model(dfs, DIM_A, group_if=group_if),
        **grid_config_application(dfs, DIM_B, practical_only=practical_only),
    )


@figure(showable=True, saveable=True, properties={"left": 0.26, "bottom": 0.28, "right": 1, "top": 0.98})
def grid_application_model(dfs, group_if: bool = False, practical_only: bool = False):
    _grid(
        **grid_config_application(dfs, DIM_A, practical_only=practical_only),
        **grid_config_model(dfs, DIM_B, group_if=group_if),
    )


@figure(showable=True, saveable=True, properties={"left": 0.21, "bottom": 0.33, "right": 1, "top": 0.98})
def grid_model_architecture(dfs, group_if: bool = False):
    _grid(
        **grid_config_model(dfs, DIM_A, group_if=group_if),
        **grid_config_architecture(dfs, DIM_B),
    )


@figure(showable=True, saveable=True, properties={"left": 0.25, "bottom": 0.27, "right": 1, "top": 0.98})
def grid_architecture_model(dfs, group_if: bool = False):
    _grid(
        **grid_config_architecture(dfs, DIM_A),
        **grid_config_model(dfs, DIM_B, group_if=group_if),
    )


@figure(showable=True, saveable=True, properties={"left": 0.21, "bottom": 0.33, "right": 1., "top": 0.98})
def grid_model_learning(dfs, group_if: bool = False):
    _grid(
        **grid_config_model(dfs, DIM_A, group_if=group_if),
        **grid_config_learning(dfs, DIM_B),
    )


@figure(showable=True, saveable=True, properties={"left": 0.26, "bottom": 0.28, "right": 1, "top": 0.98})
def grid_learning_model(dfs, group_if: bool = False):
    _grid(
        **grid_config_learning(dfs, DIM_A),
        **grid_config_model(dfs, DIM_B, group_if=group_if),
    )


@figure(showable=True, saveable=True, properties={"left": 0.25, "bottom": 0.33, "right": 1, "top": 0.98})
def grid_architecture_application(dfs, practical_only: bool = False):
    _grid(
        **grid_config_architecture(dfs, DIM_A),
        **grid_config_application(dfs, DIM_B, practical_only=practical_only),
    )


@figure(showable=True, saveable=True, properties={"left": 0.26, "bottom": 0.33, "right": 1, "top": 0.98})
def grid_application_architecture(dfs, practical_only: bool = False):
    _grid(
        **grid_config_application(dfs, DIM_A, practical_only=practical_only),
        **grid_config_architecture(dfs, DIM_B),
    )


@figure(showable=True, saveable=True, properties={"left": 0.26, "bottom": 0.34, "right": 1, "top": 0.98})
def grid_learning_application(dfs, practical_only: bool = False):
    _grid(
        **grid_config_learning(dfs, DIM_A),
        **grid_config_application(dfs, DIM_B, practical_only=practical_only),
    )


@figure(showable=True, saveable=True, properties={"left": 0.26, "bottom": 0.34, "right": 1, "top": 0.98})
def grid_application_learning(dfs, practical_only: bool = False):
    _grid(
        **grid_config_application(dfs, DIM_A, practical_only=practical_only),
        **grid_config_learning(dfs, DIM_B),
    )


@figure(showable=True, saveable=True, properties={"left": 0.26, "bottom": 0.34, "right": 1, "top": 0.98})
def grid_architecture_learning(dfs):
    _grid(
        **grid_config_architecture(dfs, DIM_A),
        **grid_config_learning(dfs, DIM_B),
    )


@figure(showable=True, saveable=True, properties={"left": 0.26, "bottom": 0.34, "right": 1, "top": 0.98})
def grid_learning_architecture(dfs):
    _grid(
        **grid_config_learning(dfs, DIM_A),
        **grid_config_architecture(dfs, DIM_B),
    )
