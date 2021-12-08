# Create plot of % missing vs. Total Error (min. Nuclear norm + avg norm of (X_obs - X_pred)?)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def line_plot(missingness, errors1, errors2=None):
    plt.figure()
    plt.plot(missingness, errors1, c='#9D2EC5')
    if errors2 is None:
        pass
    else:
        plt.plot(missingness, errors2, c='#841a06')
    plt.title("Error rate vs missingness")
    plt.xlabel("Proportion of data missing")
    plt.ylabel("Reconstruction error (Frobenius norm)")
    plt.grid(True)
    plt.show()
    # plt.savefig("output/lineplot_errors.png")


def scatter_plot(missingness, errors):
    plt.figure()
    plt.scatter(missingness, errors)
    plt.title("How is error affected by missingness?")
    plt.xlabel("proportion missing")
    plt.ylabel("error")
    plt.grid(True)
    plt.show()
    # plt.savefig("output/scatterplot_errors.png")

def plot_errors2(missingness, errors):
    sns.lineplot(x=missingness, y=errors)
    plt.show()


if __name__ == "__main__":
    # generate toy data
    proportion_missing = np.arange(0, 100, 1) / 100
    errors = np.random.randint(1, 50, size=proportion_missing.size)
    errors = [idx + error for idx, error in enumerate(errors)]
    line_plot(proportion_missing, errors)
    # scatter_plot(proportion_missing, errors)
