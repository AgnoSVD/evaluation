import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

res_dir = "."
density = {"als": 1.0, "sgd": 1.0}
thres_list = np.linspace(0, 1, 21)
line_styles = [":", "-", "-.", ]  # "--",
colors = ["green", "blue", "red", "cyan", "yellow", "pink"]
plt.figure(figsize=(7, 4))
plt.rcParams.update({'font.size': 12})
metrics = ["perf", "cost", "cost*perf"]
n_epochs = 2
for alg in ["als", "sgd"]:
    for index, metric in enumerate(metrics):
        for epoch in range(n_epochs):
            als_res_df = pd.read_csv(f"{res_dir}/{alg}_res_df_epoch_{epoch}_99_{metric}.csv")
            als_prob = [sum(i <= thres for i in als_res_df[str(density[alg])]) / len(als_res_df) for thres in thres_list]
            # if als_prob[0] > .5:
            #     als_prob[0] -= random.choice([0.2, 0.1])
            # if metric == "perf":
            #     als_prob[0] = 0.55
            plt.plot(thres_list, als_prob, line_styles[epoch % 4], label=f'{metric}' if epoch == 1 else '_nolegend_', color=colors[index])
            plt.grid(True)
            # plt.xlabel(f"{alg}")
            plt.xticks(np.linspace(0, 1, 11))
            plt.yticks(np.linspace(0, 1, 6))

    plt.legend(bbox_to_anchor=(.77, .01, .23, .1), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0.1)
    plt.ylim(0,1)
    plt.ylabel("Probality of accurate prediction", fontsize=10)
    plt.xlabel("Threshold", fontsize=10)
    plt.tight_layout(h_pad=1)
    plt.savefig(f"{res_dir}/fig_pred_acc_{alg}.png")
    plt.clf()
