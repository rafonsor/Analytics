import typing as t
from itertools import product

import pandas as pd
from matplotlib import pyplot as plt

from analytics import DataFinal
from analytics.definitions import Column, OrderedCategory
from analytics.plots.plot_utils import figure
from analytics.data.transforms import LearningTF, ReferencesTF

NO_DATA = 'No Data'


@figure(showable=True, saveable=True, tight=True)
def bar_publications_per_year(dfs: t.Dict[str, pd.DataFrame]) -> None:
    pubs = ReferencesTF.transform(dfs[DataFinal.REFERENCES.name]).groupby(Column.YEAR).count().reset_index()
    plt.grid(visible=True, which='both', axis='y')
    plt.xlabel('Year of Publication', weight=600, fontsize='large')
    plt.ylabel('Number of Publications', weight=600, fontsize='large')
    plt.bar(pubs[Column.YEAR], pubs[Column.PUBLICATION])
    plt.xlim([pubs[Column.YEAR].min() - 0.5, pubs[Column.YEAR].max() + 0.5])
    plt.ylim([0.0, ((pubs[Column.PUBLICATION].max() / 10) + 1) * 10])


@figure(showable=True, saveable=True, tight=True)
def stack_learning_algorithms(dfs: t.List[pd.DataFrame], use_fractions: bool = False, show_info: bool = True) -> None:

    df = pd.merge(
        left=ReferencesTF.transform(dfs[DataFinal.REFERENCES.name]),
        right=LearningTF.transform(dfs[DataFinal.LEARNING.name]),
        on=Column.PUBLICATION,
        how='right',
    )
    optimal = product(df[Column.LEARNING].unique(), range(df[Column.YEAR].min(), df[Column.YEAR].max() + 1))
    df = df.groupby([Column.LEARNING, Column.YEAR]).count()
    df = df.reindex(optimal, fill_value=0)

    if use_fractions:
        sums = df.groupby(Column.YEAR).sum()
        df[sums.columns] /= sums
        tofill = sums[sums[Column.PUBLICATION] == 0]
        tofill.loc[:, Column.LEARNING] = NO_DATA
        tofill.loc[:, Column.PUBLICATION] = 1
        tofill.reset_index(inplace=True)
        tofill.set_index([Column.LEARNING, Column.YEAR], inplace=True)
        df = df.append(tofill).fillna(0)

    df.reset_index(inplace=True)
    plt.xlim([df[Column.YEAR].min() - 1, df[Column.YEAR].max() + 1])

    prev = None
    for cat in OrderedCategory.LearningAlgorithms:
        idx = df[Column.LEARNING] == cat
        plt.bar(df[idx][Column.YEAR], df[idx][Column.PUBLICATION], label=cat, bottom=prev)
        prev = df[idx][Column.PUBLICATION].values if prev is None else df[idx][Column.PUBLICATION].values + prev

    idx = df[Column.LEARNING] == NO_DATA
    plt.bar(df[idx][Column.YEAR], df[idx][Column.PUBLICATION], color='lightgrey', label=NO_DATA)
    if use_fractions:
        plt.ylabel("Share per Category", weight=600, fontsize='large')
    else:
        plt.ylabel("Number of Publications", weight=600, fontsize='large')

    if show_info:
        plt.xlabel("Year of Publication", weight=600, fontsize='large')
        plt.legend().legendHandles[-1]._facecolor = (0.8274509803921568, 0.8274509803921568, 0.8274509803921568, 1.0)
