import typing as t

import pandas as pd
from matplotlib import pyplot as plt

from analytics import DataFinal
from analytics.definitions import Column
from analytics.plots.plot_utils import figure


@figure(showable=True, saveable=True, tight=True)
def pie_benchmarks(dfs: t.Dict[str, pd.DataFrame]) -> None:
    df = dfs[DataFinal.BENCHMARK.name][[Column.BENCHMARK, Column.PUBLICATION]].groupby(Column.BENCHMARK).nunique(dropna=False)
    plt.pie(df[Column.PUBLICATION], labels=df.index.values, rotatelabels=True)
    print(df[Column.PUBLICATION])
