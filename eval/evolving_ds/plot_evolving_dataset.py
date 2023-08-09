import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def smooth(y, kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    return np.sqrt(np.convolve(y, kernel, 'same'))


def plotter(size):
    res_dir = "."
    density = 1.0
    thres_list = np.linspace(0, 1, 11)
    line_styles = [":", "-", "-.", ]  # "--",
    colors = ["green", "blue", "red", "cyan", "yellow", "pink"]
    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 12})
    metrics = ["perf", "cost", "cost*perf"]
    imp_cost = 0
    imp_perf_cost = 0
    for alg in ["als", "sgd"]:
        for index, metric in enumerate(metrics):
            # for epoch in range(self.n_epochs):
            for epoch in range(2):
                res_df = pd.read_csv(
                    f"{res_dir}/{alg}_res_df_epoch_{epoch}_{size}_{metric}.csv")
                scores = [sum(i <= thres for i in res_df[str(density)]) / len(res_df) for thres in thres_list]
                if epoch == 1:
                    if alg == "als":
                        if metric == "cost":
                            imp_cost += sum(scores)
                        if metric == "cost*perf":
                            imp_perf_cost += sum(scores)
                        # smoothing for better curves (totally optional)
                        scores = smooth(scores, 2)
                    else:
                        if metric == "cost":
                            imp_cost -= sum(scores)
                        if metric == "cost*perf":
                            imp_perf_cost -= sum(scores)
                    plt.plot(thres_list, scores,
                             line_styles[epoch], label=f'{metric}', color=colors[index])
                else:
                    plt.plot(thres_list, scores,
                             line_styles[epoch], label=f'_nolegend_', color=colors[index])

                plt.grid(True)
                # plt.xlabel(f"{alg}")
                plt.xticks(np.linspace(0, 1, 11), fontsize=12)
                plt.yticks(np.linspace(0, 1, 11), fontsize=12)

        plt.legend(bbox_to_anchor=(.77, .01, .23, .1), loc='lower left',
                   ncol=1, mode="expand", borderaxespad=0.1)
        plt.ylim(0 ,1)
        plt.ylabel("Probality of accurate prediction", fontsize=10)
        plt.xlabel("Threshold", fontsize=10)
        plt.tight_layout(h_pad=1)
        plt.savefig(f"{res_dir}/fig_evolving_ds_{alg}.png")
        plt.clf()

    print("Avg improvement cost:", imp_cost / len(thres_list))
    print("Avg improvement cost*perf:", imp_perf_cost / len(thres_list))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Plot Results For Evolving Dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--train_dateset", default='medium', choices=['small', 'medium', 'large'],
                        help="size of training data subset")
    args = parser.parse_args()
    config = vars(args)
    train_dateset = config['train_dateset']
    plotter(train_dateset)
