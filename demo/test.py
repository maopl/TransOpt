import datetime

import numpy as np

if __name__ == "__main__":
    X1, X2 = np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-2, 2, 500))

    F1 = 100 * (X1**2 + X2**2)
    F2 = (X1 - 1) ** 2 + X2**2

    G1 = 2 * (X1[0] - 0.1) * (X1[0] - 0.9)
    G2 = 20 * (X1[0] - 0.4) * (X1[0] - 0.6)

    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")

    levels = np.array([0.02, 0.1, 0.25, 0.5, 0.8])
    plt.figure(figsize=(7, 5))
    CS = plt.contour(X1, X2, F1, 10 * levels, colors="black", alpha=0.5)
    CS.collections[0].set_label("$f_1(x)$")

    CS = plt.contour(X1, X2, F2, levels, linestyles="dashed", colors="black", alpha=0.5)
    CS.collections[0].set_label("$f_2(x)$")

    plt.plot(X1[0], G1, linewidth=2.0, color="green", linestyle="dotted")
    plt.plot(X1[0][G1 < 0], G1[G1 < 0], label="$g_1(x)$", linewidth=2.0, color="green")

    plt.plot(X1[0], G2, linewidth=2.0, color="blue", linestyle="dotted")
    plt.plot(
        X1[0][X1[0] > 0.6],
        G2[X1[0] > 0.6],
        label="$g_2(x)$",
        linewidth=2.0,
        color="blue",
    )
    plt.plot(X1[0][X1[0] < 0.4], G2[X1[0] < 0.4], linewidth=2.0, color="blue")

    plt.plot(np.linspace(0.1, 0.4, 100), np.zeros(100), linewidth=3.0, color="orange")
    plt.plot(np.linspace(0.6, 0.9, 100), np.zeros(100), linewidth=3.0, color="orange")

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        fancybox=True,
        shadow=False,
    )

    plt.tight_layout()
