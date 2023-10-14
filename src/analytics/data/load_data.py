#! /usr/bin/env python3
from __future__ import annotations

from enum import Enum, unique
import os

import pandas as pd


DATA_FOLDER = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
DATA_FOLDER_2021 = DATA_FOLDER + '_2021'
DATA_FOLDER_FINAL = DATA_FOLDER + '_final'


@unique
class Data(Enum):
    APPLICATIONS = 'applications.csv'
    LEARNING = "learning.csv"
    NETWORKS = "networks.csv"
    MODELS = "models.csv"
    BENCHMARK = 'benchmarking.csv'
    ARCHITECTURE = 'architectures.csv'

    @staticmethod
    def load_data(target: Data, delimiter: str = ',', source: str = DATA_FOLDER) -> pd.DataFrame:
        with open(os.path.join(source, target.value)) as fd:
            return pd.read_csv(fd, delimiter=delimiter)

    @staticmethod
    def load_all(source: str = DATA_FOLDER):
        return {
            target: Data.load_data(getattr(Data, target, source))
            for target in Data.__members__
        }


class Data2021:
    APPLICATIONS = Data.APPLICATIONS
    LEARNING = Data.LEARNING
    NETWORKS = Data.NETWORKS
    MODELS = Data.MODELS
    BENCHMARK = 'results.csv'
    ARCHITECTURE = Data.ARCHITECTURE

    @staticmethod
    def load_data(target: Data, delimiter: str = ',') -> pd.DataFrame:
        return Data.load_data(target, delimiter, source=DATA_FOLDER_2021)

    @staticmethod
    def load_all():
        return Data.load_all(source=DATA_FOLDER_2021)


@unique
class DataFinal(Enum):
    APPLICATIONS = 'Applications.csv'
    LEARNING = "Learning.csv"
    NETWORKS = "Networks.csv"
    MODELS = "Models.csv"
    BENCHMARK = 'Benchmarks.csv'
    ARCHITECTURE = 'Architectures.csv'
    REFERENCES = 'References.csv'

    @staticmethod
    def load_data(target: Data, delimiter: str = ',', source: str = DATA_FOLDER_FINAL) -> pd.DataFrame:
        print(target, source, os.path.join(source, target.value))
        with open(os.path.join(source, target.value)) as fd:
            return pd.read_csv(fd, delimiter=delimiter)

    @staticmethod
    def load_all(source: str = DATA_FOLDER_FINAL):
        return {
            key: DataFinal.load_data(target, source=source)
            for key, target in DataFinal.__members__.items()
        }
