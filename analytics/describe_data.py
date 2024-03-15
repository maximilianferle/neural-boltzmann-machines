import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, shapiro
from data.datasets import SequenceDataModule


def main():
    sg = SequenceDataModule()
    train = sg.normalize(sg._train)
    val = sg.normalize(sg._val)
    test = sg.normalize(sg._test)

    # for dist in [train, test, val]:
    #     test_quantiles(dist)

    test_quantiles(input=train, ref=val)
    test_quantiles(input=train, ref=test)
    test_quantiles(input=test, ref=val)


def test_quantiles(input: list, ref=None):
    cat_list = np.concatenate(input, axis=0)
    ref = np.random.normal(size=cat_list.shape) if ref is None else np.concatenate(ref, axis=0)
    cat_percentiles = np.percentile(cat_list, np.arange(1, 100, 3), axis=0)
    ref_percentiles = np.percentile(ref, np.arange(1, 100, 3), axis=0)
    plt.plot(cat_percentiles, ref_percentiles)
    plt.show()


if __name__ == "__main__":
    main()
