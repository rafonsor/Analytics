#! /usr/bin/env python3
import time
from contextlib import contextmanager

from matplotlib import pyplot as plt

from analytics.data.load_data import DataFinal
from analytics.data.transforms import apply_transform, ArchitecturesTF, ApplicationsTF, BenchmarkTF, LearningTF, \
    ModelsTF, ReferencesTF
from analytics.definitions import Column
from analytics.plots import *


@contextmanager
def clock(msg: str):
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    print(f'{msg}: took {toc-tic:.3f}s')


if __name__ == "__main__":
    with clock("Data load"):
        dfs = DataFinal.load_all()

    with clock("[Validation] Data transformations"):
        dfref = apply_transform(dfs, DataFinal.REFERENCES.name, ReferencesTF)
        dfmod = apply_transform(dfs, DataFinal.MODELS.name, ModelsTF)
        dfarc = apply_transform(dfs, DataFinal.ARCHITECTURE.name, ArchitecturesTF)
        dfapp = apply_transform(dfs, DataFinal.APPLICATIONS.name, ApplicationsTF)
        dflea = apply_transform(dfs, DataFinal.LEARNING.name, LearningTF)
        dfben = apply_transform(dfs, DataFinal.BENCHMARK.name, BenchmarkTF)

    with clock("[Validation] References"):
        pubs = set(dfref[Column.PUBLICATION])
        assert not (set(dfmod[Column.PUBLICATION]) - pubs), f"Unaccounted references: {DataFinal.MODELS}"
        assert not (set(dfarc[Column.PUBLICATION]) - pubs), f"Unaccounted references: {DataFinal.ARCHITECTURE}"
        assert not (set(dfapp[Column.PUBLICATION]) - pubs), f"Unaccounted references: {DataFinal.APPLICATIONS}"
        assert not (set(dflea[Column.PUBLICATION]) - pubs), f"Unaccounted references: {DataFinal.LEARNING}"
        assert not (set(dfben[Column.PUBLICATION]) - pubs), f"Unaccounted references: {DataFinal.BENCHMARK}"
        assert not (set(dfs[DataFinal.NETWORKS.name][Column.PUBLICATION]) - pubs), \
            f"Unaccounted references: {DataFinal.NETWORKS}"

    # with clock("Generate publications plots"):
    #     bar_publications_per_year(dfs, save=True)
    #     plt.close('all')
    #
    # with clock("Generate learning algorithms stack plot"):
    #     fig: plt.Figure = plt.figure(10, figsize=[6.4, 6.4])
    #     fig.subplots(2, 1, sharex=True)
    #     stack_learning_algorithms(dfs, use_fractions=True, show_info=False, show=False, fid=10, subplot=211)
    #     stack_learning_algorithms(dfs, use_fractions=False, show=False, save=True, fid=10, subplot=212)
    #     plt.close('all')
    #
    # with clock("Generate grid plots"):
    #     grid_application_model(dfs, group_if=False, practical_only=False, save=True)
    #     grid_application_model(dfs, group_if=True, practical_only=False, save=True)
    #     grid_application_model(dfs, group_if=False, practical_only=True, save=True)
    #     grid_application_model(dfs, group_if=True, practical_only=True, save=True)
    #     grid_model_application(dfs, group_if=False, practical_only=False, save=True)
    #     grid_model_application(dfs, group_if=True, practical_only=False, save=True)
    #     grid_model_application(dfs, group_if=False, practical_only=True, save=True)
    #     grid_model_application(dfs, group_if=True, practical_only=True, save=True)
    #     grid_architecture_model(dfs, group_if=False, save=True)
    #     grid_architecture_model(dfs, group_if=True, save=True)
    #     grid_model_architecture(dfs, group_if=False, save=True)
    #     grid_model_architecture(dfs, group_if=True, save=True)
    #     grid_learning_model(dfs, group_if=False, save=True)
    #     grid_learning_model(dfs, group_if=True, save=True)
    #     grid_model_learning(dfs, group_if=False, save=True)
    #     grid_model_learning(dfs, group_if=True, save=True)
    #     grid_architecture_learning(dfs, save=True)
    #     grid_learning_architecture(dfs, save=True)
    #     grid_architecture_application(dfs, practical_only=False, save=True)
    #     grid_architecture_application(dfs, practical_only=True, save=True)
    #     grid_application_architecture(dfs, practical_only=False, save=True)
    #     grid_application_architecture(dfs, practical_only=True, save=True)
    #     grid_learning_application(dfs, practical_only=False, save=True)
    #     grid_learning_application(dfs, practical_only=True, save=True)
    #     grid_application_learning(dfs, practical_only=False, save=True)
    #     grid_application_learning(dfs, practical_only=True, save=True)
    #     plt.close('all')
    #
    # with clock("Generate network size boxplots"):
    #     fig: plt.Figure = plt.figure(10, figsize=[12.8, 9.6])
    #     fig.subplots(2, 2)#, sharey=True)
    #     boxplot_network_size_per_model(dfs, show=False, fid=10, subplot=221)
    #     boxplot_network_size_per_architecture(dfs, show=False, fid=10, subplot=222)
    #     boxplot_network_size_per_learning(dfs, show=False, fid=10, subplot=223)
    #     boxplot_network_size_per_application(dfs, show=False, fid=10, subplot=224, save='boxplot_network_size')
    #     boxplot_network_size_per_model(dfs, save=True)
    #     boxplot_network_size_per_application(dfs, save=True)
    #     boxplot_network_size_per_architecture(dfs, save=True)
    #     boxplot_network_size_per_learning(dfs, save=True)
    #     plt.close('all')

    with clock("Generate network size scatterplots"):
        # scatter_network_size_over_time_per_architecture(dfs, drop_unknown=False, save=True)
        # scatter_network_size_over_time_per_architecture(dfs, drop_unknown=True, save=True)
        # scatter_network_size_over_time_per_learning(dfs, drop_unknown=False, save=True)
        # scatter_network_size_over_time_per_learning(dfs, drop_unknown=True, save=True)
        # scatter_network_size_over_time_per_application(dfs, drop_unknown=False, save=True)
        # scatter_network_size_over_time_per_application(dfs, drop_unknown=True, save=True)
        # scatter_network_size_over_time_per_application(dfs, drop_unknown=False, practical_only=True, save=True)
        # scatter_network_size_over_time_per_application(dfs, drop_unknown=True, practical_only=True, save=True)
        # scatter_network_size_over_time_per_model(dfs, drop_unknown=False, save=True)
        # scatter_network_size_over_time_per_model(dfs, drop_unknown=True, save=True)
        # scatter_network_size_over_time_per_model(dfs, drop_unknown=False, group_if=True, save=True)
        # scatter_network_size_over_time_per_model(dfs, drop_unknown=True, group_if=True, save=True)
        scatter_network_size_summary(dfs, show=True)#save='scatter_network_size_summary_trends')
        plt.close('all')

    # with clock("Generate benchmark results plots"):
    #     scatter_benchmark_results(dfs, benchmark='MNIST', save=True)
    #     scatter_benchmark_results_closeup(dfs, benchmark='MNIST', save=True, min_score=98)
    #     scatter_benchmark_results(dfs, benchmark='CIFAR-10', save=True)
    #     scatter_benchmark_results_closeup(dfs, benchmark='CIFAR-10', save=True, min_score=90)
    #     plt.figure(10, figsize=[12.8, 9.6])
    #     scatter_benchmark_results_multiple(
    #         dfs, what=['CIFAR-100', 'ImageNet', 'DVS128', 'IRIS', 'BREAST CANCER', 'TI46'], fid=10, save=True)
    #     plt.close('all')
