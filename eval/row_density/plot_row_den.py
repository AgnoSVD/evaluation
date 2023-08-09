import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator


def smooth(x, y, kernel_size):
    # kernel = np.ones(kernel_size) / kernel_size
    # return x, np.sqrt(np.convolve(y, kernel, 'same'))
    return x, savgol_filter(y, kernel_size, 2)
    # x_y_spline = PchipInterpolator(x, y)
    # x = np.linspace(min(x), max(x), kernel_size)
    # return x, x_y_spline(x)


def plotter(n_jobs):
    res_dir = "."

    colors = ["green", "blue", "red", "cyan", "yellow", "pink"]

    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 12})

    metrics = ["perf", "cost", "cost*perf"]
    # density_list = np.linspace(0.2, 1, 9)
    # density_list = np.linspace(0.2, 1, 5)
    density_list = np.linspace(0.2, 1, 81)
    imp_perf = 0
    imp_cost = 0
    imp_perf_cost = 0
    for alg in ["als", "sgd"]:
        for index, metric in enumerate(metrics):
            scores = []
            epoch = 1
            # res_dir = res_dir if alg=="als" else "../best_sum"
            res_df = pd.read_csv(f"{res_dir}/{alg}_res_df_epoch_{epoch}_{n_jobs}_{metric}.csv")

            for ind, density in enumerate(density_list):
                score = sum(i <= 0.15 for i in res_df[str(density)]) / len(res_df)
                if alg == "als":
                    # if ind == 0 or ind > 3:
                    scores.append(score)
                    # else:
                    #     if score > scores[ind - 1]:
                    #         scores.append(score)
                    #     else:
                    #         scores.append(scores[ind - 1] - 0.02)
                else:
                    scores.append(score)
            x, y = density_list, scores
            # smoothing for better curves (totally optional)
            x, y = smooth(x, y, 51)
            if alg == "als":
                plt.plot(x, y, "-", label=f'{metric}', color=colors[index])
                if metric == "perf":
                    imp_perf += sum(scores)
                if metric == "cost":
                    imp_cost += sum(scores)
                if metric == "cost*perf":
                    imp_perf_cost += sum(scores)
            else:
                plt.plot(x, y, "--", label=f'_nolegend_', color=colors[index])
                if metric == "perf":
                    imp_perf -= sum(scores)
                if metric == "cost":
                    imp_cost -= sum(scores)
                if metric == "cost*perf":
                    imp_perf_cost -= sum(scores)
            plt.grid(True)
            plt.xticks(np.linspace(0.2, 1, 9))
            plt.yticks(np.linspace(0, 1, 11))

    plt.legend(bbox_to_anchor=(.77, .01, .23, .1), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0.1)
    plt.ylabel("Probality of predicting 15% of optimum", fontsize=10)
    plt.xlabel("Fraction of configs profiled per training job", fontsize=10)
    plt.tight_layout(h_pad=1)
    plt.savefig(f"./fig_row_den_{n_jobs}.png")
    plt.clf()

    print("Avg improvement perf:", imp_perf / len(density_list))
    print("Avg improvement cost:", imp_cost / len(density_list))
    print("Avg improvement cost*perf:", imp_perf_cost / len(density_list))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="System State",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--state", default="steady_state", choices=['steady_state', 'cold_start'],
                        help="benchmark scenario")
    args = parser.parse_args()
    config = vars(args)
    state = config['state']
    if state == "steady_state":
        plotter(99)
    if state == "cold_start":
        plotter(40)
