# Create plot of % missing vs. Total Error (min. Nuclear norm + avg norm of (X_obs - X_pred)?)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def line_plot(missingness, errors1, errors2=None, save=False, title="", xlabel="", ylabel=""):
    plt.figure()
    plt.plot(missingness, errors1, c='#9D2EC5')
    if errors2 is None:
        pass
    else:
        plt.plot(missingness, errors2, c='#33bb55')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend([errors1.name, errors2.name])
    if save:
        plt.savefig(f"output/{title.replace(' ', '_')}.png")
    else:
        plt.show()

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
