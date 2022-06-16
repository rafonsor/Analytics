import typing as t

import pandas as pd
import seaborn as sb

from analytics.plots.plot_utils import figure


@figure(showable=True, saveable=True, tight=True)
def spider_architectures(dfs: t.Dict[str, pd.DataFrame]) -> None:
    # for proportion of architectures in publications (rather than networks as in pie charts)
    sb.FacetGrid()


@figure(showable=True, saveable=True, tight=True)
def spider_models(dfs: t.Dict[str, pd.DataFrame]) -> None:
    # for proportion of models in publications (rather than networks as in pie charts)
    sb.FacetGrid()

