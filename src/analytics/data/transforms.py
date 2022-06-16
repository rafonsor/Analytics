# import functools
import re
import typing as t
from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd

from analytics.definitions import Column, Model, Architecture, UNKNOWN_LABEL

ALL_IF = ['LIF', 'IF (Other)', 'SRM']

DEFAULT_YEAR_JITTER = 0.2  # std for gaussian noise applied to Year to spread data points over X axis


def allow_inplace(fn: callable) -> callable:
    # @functools.wraps(fn)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        # kwargs.setdefault('inplace', False)
        return fn(df if kwargs.pop('inplace', False) else df.copy(), *args, **kwargs)
    return wrapper


class TransformI(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def transform(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform data from a survey category"""
        pass


def apply_transform(dfs: dict, key: str, transform: t.Type[TransformI], **kwargs) -> pd.DataFrame:
    assert key in dfs, f"Missing {key} in provided data frames"
    return transform.transform(dfs[key], **kwargs)


class ModelsTF(TransformI):

    Models = [
        Model(0, '#ea4335', 'Leaky Integrate-and-Fire', 'LIF', ['LIF', 'GLIF']),
        Model(1, '#f4665b', 'Integrate-and-Fire (Other)', 'IF (Other)', ['IF', 'CUBA', 'EIF', 'AEIF', 'QIF']),
        Model(2, '#f07b72', 'Spike Response Model', 'SRM', ['SRM']),
        Model(3, '#cfcf15', 'Izhikevich', 'Izhikevich', ['IZ']),  # #f1f113
        Model(4, '#34a853', 'Hodgkin-Huxley', 'HH', ['HH']),
        Model(5, '#46bdc6', 'Multi-Compartmental', 'Multi-Comp.', ['DS', 'Gap junctions', 'Compartmental', 'Nonlinear Dendrites']),
        Model(7, '#434343', 'Unknown', 'Unknown', ['-']),
    ]
    ModelsIF = Model(1, '#ea4335', 'Integrate-and-Fire (All)', 'IF (All)', [name for m in Models if m.short in ALL_IF for name in m.models])
    ModelsOther = Model(6, '#864ea7', 'Other Models', 'Other Models', [name for m in Models for name in m.models])
    
    @staticmethod
    @allow_inplace
    def transform(df: pd.DataFrame, short_name: bool = True, group_if: bool = False) -> pd.DataFrame:
        """Group specific models into their corresponding meta class
    
        :param df: data frame with spiking models
        :param short_name: use abbreviated model class name
        :param group_if: group all Integrate-and-Fire variants under the same model class
        :return: copy of input data frame with concrete models grouped into parent model class
        """
        assert Column.NEURON_MODEL in df.columns, "Expecting data frame with model information"
        dfout = df.copy()
        ds = dfout[Column.NEURON_MODEL]
        ds[~ds.isin(ModelsTF.ModelsOther.models)] = ModelsTF.ModelsOther.short if short_name else ModelsTF.ModelsOther.name
        models = ModelsTF.Models if not group_if else [m for m in ModelsTF.Models if m.short not in ALL_IF] + [ModelsTF.ModelsIF]
        for info in models:
            ds[ds.isin(info.models)] = info.short if short_name else info.name
        return dfout


class ArchitecturesTF(TransformI):
    Architectures = [
        Architecture(id=0, name="Recurrent", columns=["Feedforward/Recurrent"], anything=[], notin=['Convolutional', 'Residual', 'Boltzmann', 'Other', 'ESN', 'LSM', 'Other Reservoir'], common=['MLP', 'RNN']),
        Architecture(id=1, name="Convolutional", columns=["Convolutional"], anything=[], notin=['Residual', 'Boltzmann', 'Other', 'ESN', 'LSM', 'Other Reservoir'], common=['CNN', 'HMAX']),
        Architecture(id=2, name="Residual", columns=["Residual"], anything=[], notin=['Boltzmann', 'Other', 'ESN', 'LSM', 'Other Reservoir'], common=['Residual']),
        Architecture(id=3, name="Boltzmann", columns=["Boltzmann"], anything=[], notin=[], common=['RBM', 'DBN']),
        Architecture(id=4, name="Autoencoder", columns=["Autoencoder"], anything=[], notin=[], common=['VAE', 'AE']),
        Architecture(id=5, name="Other (Layer)", columns=[], anything=['Other'], notin=['Columnar']),
        Architecture(id=6, name="LSM", columns=['LSM'], anything=[], notin=[], common=['LSM']),
        Architecture(id=7, name="Other (Reservoir)", columns=['ESN'], anything=['Other Reservoir'], notin=['LSM', 'Columnar'], common=['ESN', 'NeuCube', 'Reservoir']),
        Architecture(id=8, name="Columnar", columns=['Columnar'], anything=[], notin=[], common=['HTM', 'Column']),
    ]

    @staticmethod
    @allow_inplace
    def _separate_autoencoders(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df['Other'].isin(['autoencoder', 'VAE']), ['Autoencoder', 'Other']] = ('s', np.NaN)
        return df

    @staticmethod
    @allow_inplace
    def _separate_microcolumns(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df['Other'] == 'Column', ['Columnar', 'Other']] = ('s', np.NaN)
        df.loc[df['Other Reservoir'] == 'Column', ['Columnar', 'Other Reservoir']] = ('s', np.NaN)
        return df

    @staticmethod
    @allow_inplace
    def _clean_others(df: pd.DataFrame) -> pd.DataFrame:
        def clean(x: t.Any) -> t.Any:
            if x is np.NaN:
                return x
            for exclude in ['continuous', 'ANN', 'HMM', 'KNN']:
                if exclude in x:
                    return np.NaN
            return x
        df['Other'] = df['Other'].apply(lambda x: clean(x))
        return df

    @staticmethod
    @allow_inplace
    def transform(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dftemp = ArchitecturesTF._clean_others(df, inplace=True)
        dftemp = ArchitecturesTF._separate_microcolumns(dftemp, inplace=True)
        dftemp = ArchitecturesTF._separate_autoencoders(dftemp, inplace=True)
        dfout = pd.DataFrame([])
        for info in ArchitecturesTF.Architectures:
            idcs = False
            for col in info.columns:
                idcs |= dftemp[col] == 's'
            for col in info.anything:
                idcs |= dftemp[col].notna()
            for col in info.notin:
                idcs &= dftemp[col] != 's'
            if sum(idcs):
                dfout = pd.concat([dfout, pd.DataFrame(dftemp[idcs][Column.PUBLICATION].apply(lambda x: (x, info.name)).tolist())])
        dfout.columns = [Column.PUBLICATION, Column.ARCHITECTURE]
        return dfout


class ReferencesTF(TransformI):

    @staticmethod
    @allow_inplace
    def transform(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.drop(columns=set(df.columns) - {Column.PUBLICATION}, inplace=True)
        df[Column.YEAR] = df[Column.PUBLICATION].transform(lambda p: int(re.search(r"\d{4}", p).group()))
        return df


class LearningTF(TransformI):

    @staticmethod
    @allow_inplace
    def transform(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.drop(columns=set(df.columns) - {Column.PUBLICATION, Column.LEARNING}, inplace=True)
        df[Column.LEARNING] = df[Column.LEARNING].apply(lambda l: re.sub(r'^[a-z]', lambda x: x.upper(), l))
        df.loc[df[Column.LEARNING] == 'Dynamical', Column.LEARNING] = 'Numerical Optim.'
        return df


class ApplicationsTF(TransformI):

    @staticmethod
    @allow_inplace
    def transform(df: pd.DataFrame, practical_only: bool = False) -> pd.DataFrame:
        df.loc[df['Focus'] == 'benchmarks', 'Area'] = 'Benchmark'
        df.loc[df['Area'].isin(['Cognitive', 'Reinforcement Learning']), 'Area'] = 'Other'
        df.drop(columns=['Focus', 'Notes'], inplace=True)
        if practical_only:
            df = df[~df[Column.APPLICATION].isin(['Benchmark', 'Simulation'])]
        return df


class BenchmarkTF(TransformI):
    Models = ModelsTF.Models
    ModelsIF = ModelsTF.ModelsIF
    ModelsOther = ModelsTF.ModelsOther
    Architectures = ArchitecturesTF.Architectures

    LearningCol = 'Learning Algorithm'
    ModelCol = 'Neuron'
    ArchitectureCol = 'Architecture'

    @staticmethod
    @allow_inplace
    def transform(df: pd.DataFrame, manual_jitter: bool = False, **kwargs) -> pd.DataFrame:
        if manual_jitter:
            df[Column.YEAR] = df[Column.YEAR].apply(lambda x: np.random.normal(x, DEFAULT_YEAR_JITTER))
        df.loc[df[BenchmarkTF.LearningCol] == 'Dynamical', BenchmarkTF.LearningCol] = 'Numerical Optim.'
        for (column, func) in [
            (BenchmarkTF.ModelCol, BenchmarkTF._substitute_model),
            (BenchmarkTF.ArchitectureCol, BenchmarkTF._substitute_architecture)
        ]:
            df[column] = df[column].apply(func)
        return df

    @staticmethod
    def _substitute_model(model: str) -> str:
        if model == "LIF & IF":
            model = "LIF"
        for info in BenchmarkTF.Models:
            if model in info.models:
                return info.short
        if not model:
            return UNKNOWN_LABEL
        return BenchmarkTF.ModelsOther.short

    @staticmethod
    def _substitute_architecture(arc: str) -> str:
        default_case: t.Optional[Architecture] = None
        for info in ArchitecturesTF.Architectures:
            if not info.common:
                default_case = info
                continue
            for name in info.common:
                if name in arc:
                    return info.name
        return default_case.name if default_case else UNKNOWN_LABEL
