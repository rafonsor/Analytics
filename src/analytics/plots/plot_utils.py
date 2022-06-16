import typing as t
from os.path import join

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analytics import DataFinal
from analytics.definitions import Label, Column, UNKNOWN_LABEL
from analytics.data.transforms import TransformI

OUTPUT_DIR = '/phd/analytics/output'


def stringify_kwargs(kwargs: t.Dict[str, t.Any]) -> str:
    """Convert bool and string keyword arguments into an underscore-chained string

    Example:
        func(FlagA=False, Flag_B=True, FlagC='Value A_OK') => "Flag-B_Value-A-OK"
    """
    options = []
    for k, v in kwargs.items():
        if not v:
            continue
        if isinstance(v, bool):
            options.append(k.replace('_', '-'))
        elif isinstance(v, str):
            options.append(v.replace('_', '-').replace(' ', '-'))
    return '_'.join(options)


def figure(
        *,
        showable: bool = False,
        saveable: bool = False,
        tight: bool = False,
        properties: t.Dict[str, t.Any] = None,
) -> callable:

    def decorator(fn: callable) -> callable:

        def wrapper(*args, **kwargs) -> None:
            plt.figure(kwargs.pop('fid', None), layout='tight' if tight else None)
            plt.subplot(kwargs.pop('subplot', 111))
            show = kwargs.pop('show', False) if showable else False
            save = kwargs.pop('save', False) if saveable else False
            modifiers = stringify_kwargs(kwargs)
            fn(*args, **kwargs)
            if modifiers or not properties:
                plt.tight_layout()
            else:
                plt.subplots_adjust(**properties)
            if save:
                filename = f'{save if isinstance(save, str) else fn.__name__}{f"_{modifiers}" if modifiers else ""}.png'
                path = join(OUTPUT_DIR, filename)
                plt.savefig(path, dpi=300)
            if show:
                plt.show()

        return wrapper

    return decorator


def mark_unknown(fn: callable) -> callable:
    """Decorator to assign data points undefined in terms of `column` to an "unknown" class"""
    def wrapper(df, *args, column: str, labels: t.List[str], **kwargs) -> None:
        if kwargs.pop('drop_unknown', False):
            df.dropna(inplace=True)
        elif df[column].isna().sum():
            df.loc[df[column].isna(), column] = UNKNOWN_LABEL
            if UNKNOWN_LABEL not in labels:
                labels = labels + [UNKNOWN_LABEL]
        fn(df, *args, column=column, labels=labels, **kwargs)
    return wrapper


def network_size_in_log_scale(highest) -> None:
    plt.yscale("log")
    plt.grid(axis="y")
    exponent = int(np.log10(highest)) + 1
    plt.ylim([1, pow(10, exponent)])
    plt.yticks([pow(10, e) for e in range(exponent + 1)])
    plt.ylabel(Label.NETWORK_SIZE, weight=600, fontsize='large')


def merge_network_size_with(
        dfs: t.Dict[str, pd.DataFrame],
        other: str,
        tf: t.Type[TransformI],
        **kwargs
) -> pd.DataFrame:
    return pd.merge(
        left=dfs[DataFinal.NETWORKS.name],
        right=tf.transform(dfs[other], **kwargs),
        on=Column.PUBLICATION,
        how='left',
    )
