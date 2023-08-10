import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

def smooth(x,y, kernel_size):
    # pushes curve up
    # kernel = np.ones(kernel_size) / kernel_size
    # y = np.sqrt(np.convolve(y, kernel, 'same'))

    y = savgol_filter(y, kernel_size, 3)

    # x_y_spline = PchipInterpolator(x, y)
    # x = np.linspace(min(x), max(x), kernel_size)
    # y = x_y_spline(x)

    return x, y

plt.figure(figsize=(7, 4))
plt.rcParams.update({'font.size': 16})
colors = ["green", "blue", "red", "cyan", "yellow", "pink"]
n_job_list = range(2, 100)  # [12, 24, 40, 48, 78, 99]
imp_cost = 0
imp_cost_perf = 0
for alg in ["als", "sgd"]:

    score_ind = 0 if alg == "als" else 1
    for index, metric in enumerate(["perf", "cost", "cost*perf"]):
        svd_scores = []
        conf_fracs = []
        for n_jobs in n_job_list:  #
            if alg == "als":
                file = open(f"./comparison_{n_jobs}_{metric}.txt")
            else:
                file = open(f"./comparison_{n_jobs}_{metric}.txt")
            scores = [float(x) for x in file.read().split(',')]
            svd_scores.append(scores[score_ind])
            conf_fracs.append(n_jobs / 16)
        x, y = conf_fracs, svd_scores
        # smoothing for better curves (totally optional)
        x, y = smooth(x, y, 31)
        if alg == "als":
            plt.plot(x, y, "-", label=f'{metric}', color=colors[index], linewidth=2)
            if metric == "cost":
                imp_cost += sum(svd_scores)
            if metric == "cost*perf":
                imp_cost_perf += sum(svd_scores)
        else:
            plt.plot(x, y, "--", label=f'_nolegend_', color=colors[index], linewidth=2)
            if metric == "cost":
                imp_cost -= sum(svd_scores)
            if metric == "cost*perf":
                imp_cost_perf -= sum(svd_scores)
        plt.grid(True)
        plt.xticks(np.linspace(0, 6, 7))
        plt.yticks(np.linspace(0, 1, 6))

plt.legend(bbox_to_anchor=(.73, .01, .27, .1), loc='lower left',
           ncol=1, mode="expand", borderaxespad=0.1)
plt.ylabel("Probability of predicting within T=15%", fontsize=12)
plt.xlabel("Ratio of num training jobs vs num of configs", fontsize=14)
plt.tight_layout(h_pad=1)
plt.savefig(f"./fig_n_jobs.png")
plt.clf()

print("Avg improvement cost:", imp_cost / len(n_job_list))
print("Avg improvement cost*perf:", imp_cost_perf / len(n_job_list))
