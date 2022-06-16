import typing as t
from dataclasses import dataclass

UNKNOWN_LABEL = 'Unknown'


class Column:
    APPLICATION = "Area"
    ARCHITECTURE = "Architecture"
    BENCHMARK = "Benchmark"
    NEURON_MODEL = "Model"
    NETWORK_SIZE = "Size"
    PUBLICATION = "Reference"
    LEARNING = 'Category'
    YEAR = 'Year'


class Label:
    APPLICATION = "Area of Application"
    ARCHITECTURE = "Network Architecture"
    BENCHMARK = "Benchmark Name"
    NEURON_MODEL = "Model Class"
    NETWORK_SIZE = "Network Size (#Neurons)"
    LEARNING = 'Learning Algorithm'
    YEAR = 'Year of Publication'


class OrderedCategory:
    Applications = ['Simulation', 'Benchmark', 'Robotics', 'Computer Vision', 'Medical', 'Signal Processing', 'Other']
    ApplicationsPractical = ['Robotics', 'Computer Vision', 'Medical', 'Signal Processing', 'Other']
    LearningAlgorithms = ['Plasticity', 'Backpropagation', 'Energy', 'Numerical Optim.', 'Evolutionary', 'Other']
    Architectures = [
        "Recurrent", "Convolutional", "Residual", "Boltzmann", "Autoencoder", "Other (Layer)", "LSM",
        "Other (Reservoir)", "Columnar"
    ]
    Models = ['LIF', 'IF (Other)', 'SRM', 'Izhikevich', 'HH', 'Multi-Comp.', 'Other Models', 'Unknown']
    ModelsSingleIF = ['IF (All)', 'Izhikevich', 'HH', 'Multi-Comp.', 'Other Models', 'Unknown']


@dataclass(frozen=True)
class Architecture:
    id: int
    name: str
    columns: t.List[str]
    anything: t.List[str]
    notin: t.List[str]
    color: str = None
    common: t.List[str] = None


@dataclass(frozen=True)
class Model:
    id: int
    color: str
    name: str
    short: str
    models: t.List[str]


@dataclass(frozen=True)
class Benchmark:
    name: str
    sota: float
    holder: str


SOTA = {
    benchmark.name: benchmark
    for benchmark
    in [
        Benchmark('MNIST', sota=99.79, holder="Wan2013"),
        Benchmark('CIFAR-10', sota=99.4, holder="Kolesnikov2019"),
    ]
}
