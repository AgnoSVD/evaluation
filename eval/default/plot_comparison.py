from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

res_dir = '.'
columns = ['SVD-ALS', 'Selecta', 'Sizeless', 'Default', 'Max cost', 'Min cost']
metrics = ["perf", "cost", "cost*perf"]
patterns = ["*", "\\", "+", "/", "O", "x"]

n_jobs = 99
x = np.arange(len(metrics))
width = 0.12
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(7, 5))

data = []
for metric in metrics:
    with open(f"{res_dir}/comparison_{n_jobs}_{metric}.txt", "r") as file:
        arr = [round(float(x), 2) for x in file.read().split(',')]
        data.append(arr)

data_df = pd.DataFrame(data=data, columns=columns, index=metrics)

pos = x - width*2
for index, col in enumerate(columns):
    rects = ax.bar(pos, data_df[col], width, label=col, hatch = patterns[index])
    # ax.bar_label(rects, padding=5, rotation=90)
    pos+=width


ax.set_ylabel('probabily to predict within 15% of optimum')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=0, fontsize=12)
ax.set_yticks(np.linspace(0,1,11))
ax.set_yticklabels(["%.1f" % x for x in np.linspace(0,1,11)], fontsize=12)
ax.grid(axis='y')
ax.set_axisbelow(True)
ax.legend(ncol=3)


plt.tight_layout()
plt.savefig(f'{res_dir}/fig-comparison.png')