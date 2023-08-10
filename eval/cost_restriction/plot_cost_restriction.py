import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


res_dir = "."
density = 1.0
thres_list = np.linspace(0, 1, 11)
colors = ["C0", "C1", "C2", "C3", "C4"]
plt.figure(figsize=(7, 4))
plt.rcParams.update({'font.size': 16})
metrics = ["0.9", "0.5", "0.1", "0.05", "0.01"]
for alg in ["als", "sgd"]:
    for index, metric in enumerate(metrics):
        epoch = 1
        als_res_df = pd.read_csv(f"{res_dir}/{alg}_res_df_epoch_{epoch}_{metric}.csv")
        als_prob = [sum(i <= thres for i in als_res_df[str(density)]) / len(als_res_df) for thres in thres_list]
        als_prob[0] = min(0.1, als_prob[0])
        if alg=="als":
            plt.plot(thres_list, als_prob, "-", label=f'cost restriction: {metric}', color=colors[index],
                     linewidth=2)
        else:
            plt.plot(thres_list, als_prob, "--", label=f'_nolegend_', color=colors[index],
                     linewidth=2)
        plt.grid(True)
        # plt.xlabel(f"{alg}")
        plt.xticks(np.linspace(0, 1, 11))
        plt.yticks(np.linspace(0, 1, 6))

plt.legend(bbox_to_anchor=(.54, .01, .46, .1), loc='lower left',
            ncol=1, mode="expand", borderaxespad=0.1)
plt.ylabel("Probability of accurate prediction", fontsize=14)
plt.xlabel("Threshold", fontsize=14)
plt.tight_layout(h_pad=1)
plt.savefig(f"{res_dir}/fig_cost_restriction.png")
plt.clf()
