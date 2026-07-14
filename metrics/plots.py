import os
import matplotlib.pyplot as plt

from metrics.moving_statistics import moving_mean_and_var
from config.paths import COMP_DIR


def plot_series_comparison(
    baseline,
    noisy,
    a2c,
    ylabel,
    title,
    filename,
    window=20,
):
    x_b, mb, vb = moving_mean_and_var(baseline, window)
    x_n, mn, vn = moving_mean_and_var(noisy, window)
    x_a, ma, va = moving_mean_and_var(a2c, window)

    plt.figure(figsize=(8, 5))
    plt.plot(x_b, mb, label="Baseline DQN")
    plt.plot(x_n, mn, label="Noisy DQN")
    plt.plot(x_a, ma, label="A2C")
    plt.xlabel("Episode")
    plt.ylabel(f"{ylabel} (moving avg, window={window})")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(COMP_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot → {out_path}")
