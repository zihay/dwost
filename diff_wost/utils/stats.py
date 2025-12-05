from dataclasses import dataclass

import numpy as np


@dataclass
class Statistics:
    def mean(self, x):
        _mean = 0.0
        _means = np.zeros(len(x))
        for i in range(len(x)):
            delta = x[i] - _mean
            _mean += delta / (i + 1)
            _means[i] = _mean
        return _means

    def m2(self, x):
        _mean = 0.0
        _means = np.zeros(len(x))
        m2 = 0.0
        m2s = np.zeros(len(x))
        for i in range(len(x)):
            delta = x[i] - _mean
            _mean += delta / (i + 1)
            _means[i] = _mean
            m2 += delta * (x[i] - _mean)
            m2s[i] = m2
        return m2s

    def var(self, x):
        m2s = self.m2(x)
        return m2s / (np.arange(len(x)) + 1)

    def ci(self, x):
        vars = self.var(x)
        return 1.96 * np.sqrt(vars) / np.sqrt(np.arange(len(x)) + 1)


def plot_ci(data, **kwargs):
    import matplotlib.pyplot as plt

    stats = Statistics()
    m = stats.mean(data)
    ci = stats.ci(data)
    plt.plot(m, **kwargs)
    plt.fill_between(np.arange(len(m)), m - ci, m + ci, alpha=0.3)


def plot_lorenz_curve(data):
    import matplotlib.pyplot as plt

    data = np.sort(data)
    y = data.cumsum() / data.sum()
    x = np.arange(len(data)) / len(data)
    plt.plot(x, y)
