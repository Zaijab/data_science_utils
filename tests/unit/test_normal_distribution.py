import matplotlib.pyplot as plt
import numpy as np
import os
import pytest


@pytest.mark.parametrize(
    "mu,sigma,filename",
    [
        (0, 1, "normal_distribution_test_figure_1.png"),
        (2, 0.5, "normal_distribution_test_figure_2.png"),
    ],
)
def test_normal_plot(mu, sigma, filename, ROOT_DIR):
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(f"N({mu}, {sigma}²)")
    ax.grid(True)

    os.makedirs(ROOT_DIR / "plots", exist_ok=True)
    save_path = os.path.join(ROOT_DIR / "plots", filename)
    fig.savefig(save_path)
    plt.close(fig)

    assert os.path.exists(save_path)
