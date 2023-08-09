import json
import random
import time
import warnings
from copy import copy
from math import floor, ceil

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, Prediction, accuracy, SVDpp, KNNBaseline
from tqdm import tqdm

from als_recommendation import ALS

import matplotlib.pyplot as plt


class Benchmark:
    def __init__(self, metric, train_size, dataset):
        self.data_file = "./dataset/dataset_full.csv"
        self.train_data_file = f"./dataset/dataset_{dataset}.csv"
        self.worker_conf = "worker_conf.json"
        self.config_info = json.load(open(self.worker_conf))
        self.config_info = dict(
            zip([str(int(x) - 1).zfill(2) for x in self.config_info.keys()], self.config_info.values()))
        self.n_configs = len(self.config_info)
        self.cpu_rate = 0.0000001
        self.memory_rate = 0.000000001
        self.empty_val = 99
        self.ref_confs = [21, 6]
        if metric == "perf":
            self.max_value = 300102.0
            self.min_value = 1.3333333333333333
        elif metric == "cost":
            self.max_value = 4.9762558480000007
            self.min_value = 7.12e-07
        else:
            self.max_value = 1493193.5760223232
            self.min_value = 1.424e-06

        self.metric = metric
        self.train_df, _, _ = self.init_data(self.train_data_file)
        self.ref_df, self.ref_confs, self.config_info = self.init_data(self.data_file)
        self.reader = Reader(rating_scale=(0, 1))
        self.dataset = Dataset(reader=self.reader)
        self.n_epochs = 2
        self.n_jobs = len(self.ref_df.job_id.unique())
        self.job_id_list = range(int(len(self.ref_df.job_id.unique())))
        # RMSE invalid if multiple density
        self.density_list = [1.0]
        self.train_size = train_size
        self.evolving_dataset = dataset
        self.test_job_id_list = self.ref_df[~self.ref_df['job_id'].isin(self.train_df['job_id'])]['job_id'].unique()
        # self.thres_list = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15,0.175, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
        self.thres_list = np.linspace(0, 1, 11)
        self.als_train_time = 0
        self.sgd_train_time = 0
        self.total_trains = 0

    def normalize(self, value):
        value = (value - self.min_value) / (self.max_value - self.min_value)
        return value

    def denormalize(self, value):
        value = (value * (self.max_value - self.min_value)) + self.min_value
        return value

    def init_data(self, data_file):
        data_df = pd.read_csv(data_file, header=None, names=['job_id', 'conf_id', 'cost', "timestamp"])
        data_df = data_df[data_df['conf_id'].isin([int(x) for x in self.config_info.keys()])]

        max_value = 300102.0  # data_df['cost'].max()
        min_value = 1.3333333333333333  # data_df['cost'].min()

        def normalize(value):
            value = (value - min_value) / (max_value - min_value)
            return value

        data_df['cost'] = data_df['cost'].apply(normalize)
        # print(max_value)
        # print(min_value)

        conf_rename = {}
        conf_info = {}
        for index, conf_id in enumerate(self.config_info):
            conf_rename[int(conf_id)] = index
            conf_info[index] = self.config_info[conf_id]

        data_df.replace({'conf_id': conf_rename}, inplace=True)
        ref_confs = [conf_rename[x] for x in self.ref_confs]
        # self.config_info = conf_info
        return data_df, ref_confs, conf_info

    def get_weights(self, nodes_per_conf, n_samples):
        weights = [max(n_samples - x, 1e-10) / n_samples for x in nodes_per_conf]
        weights = [x / sum(weights) for x in weights]  # np.exp(weights - np.max(weights))
        return weights

    # def update_nodes_per_conf(self, nodes_per_conf, sampled_rows):
    #     for conf_id in sampled_rows.index:
    #         nodes_per_conf[conf_id] += 1
    #     return nodes_per_conf
    #
    # def sample_np(self, non_ref_rows, weights, seed, n_samples):
    #     np.random.seed(seed)
    #     non_ref_rows = non_ref_rows.sort_values(by=['conf_id'], ascending=True).reset_index(drop=True)
    #     sampled_indices = np.random.choice(non_ref_rows.index, n_samples, p=weights, replace=False)
    #     sampled_rows = non_ref_rows[non_ref_rows.index.isin(sampled_indices)]
    #     return sampled_rows

    def get_avg_difference(self, v_2, v_1):
        v_2 = self.denormalize(v_2)
        v_1 = self.denormalize(v_1)
        value = abs(v_2 - v_1) / ((v_2 + v_1) / 2)
        return value

    def df_to_mat(self, df):
        temp_df = df.drop("timestamp", axis=1)
        matrix = np.array(temp_df.pivot(index='job_id', columns='conf_id', values='cost').fillna(self.empty_val))
        return matrix

    def run_als(self, train_df, test_job_id, known_confs, n_epoch):
        def get_rmse(true_list, pred_list):
            sum_sqr_err = 0
            total_pred = 0
            for conf_id, (cost, est) in enumerate(zip(true_list, pred_list)):
                # print(test_job_id, conf_id, '\t', cost, '\t', est)
                if conf_id not in self.ref_confs:
                    cost = self.denormalize(cost)
                    est = self.denormalize(est)
                    sum_sqr_err += (cost - est) ** 2
                    total_pred += 1
            rmse = (sum_sqr_err / total_pred) ** 0.5
            return rmse

        train_X = self.df_to_mat(train_df)
        ref_X = self.df_to_mat(self.ref_df)
        temp = ref_X[test_job_id][known_confs]
        test_job_id = train_X.shape[0]
        train_X = np.insert(train_X, test_job_id, np.full(self.n_configs, self.empty_val), axis=0)
        train_X[test_job_id][known_confs] = temp
        if self.metric=='cost*perf':
            model = ALS(X=train_X, K=15, lamb_u=0.005, lamb_v=.005, none_val=self.empty_val, max_epoch=10)
        elif self.metric=='cost':
            model = ALS(X=train_X, K=15, lamb_u=0.005, lamb_v=.05, none_val=self.empty_val, max_epoch=10)
        else:
            model = ALS(X=train_X, K=10, lamb_u=0.05, lamb_v=.0005, none_val=self.empty_val, max_epoch=5)
        start = time.time()
        model.train()

        self.als_train_time += (time.time() - start)

        est_matrix = model.get_est_matrix()

        # rmse_perf = get_rmse(ref_X[test_job_id], est_matrix[test_job_id])
        # print("RMSE using ALS perf:", rmse_perf)
        def apply_objective(runtime_list):

            cost_list = []
            for conf_id, runtime in enumerate(runtime_list):
                runtime_real = self.denormalize(runtime)
                config = self.config_info[int(conf_id)]
                if self.metric == "perf":
                    cost = runtime_real
                elif self.metric == "cost":
                    cost = (config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * runtime_real
                else:
                    cost = (config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * runtime_real ** 2

                cost = self.normalize(cost)
                cost_list.append(cost)
            return np.array(cost_list)

        pred_list = apply_objective(est_matrix[test_job_id])
        true_list = apply_objective(ref_X[test_job_id])
        rmse_obj = get_rmse(true_list, pred_list)
        # print("RMSE using ALS Obj:", rmse_obj)

        pred_conf_id = np.argmin(pred_list, axis=0)
        true_conf_id = np.argmin(true_list, axis=0)

        cost_diff = self.get_avg_difference(true_list[pred_conf_id],
                                            true_list[true_conf_id])
        return cost_diff, pred_conf_id, rmse_obj

    def run_sgd(self, train_df, test_job_id, known_confs, n_epoch):

        def get_rmse(predictions):
            denormalized = []
            for x in predictions:
                if x.iid not in self.ref_confs:
                    r_ui = self.denormalize(x.r_ui)
                    est = self.denormalize(x.est)
                    denormalized.append(Prediction(x.uid, x.iid, r_ui, est, x.details))
            rmse = accuracy.rmse(denormalized, verbose=False)
            return rmse

        test_job_id = test_job_id  # test_job_id is not an index
        train_nodes = pd.concat([train_df[train_df['job_id'] != test_job_id],
                                 self.ref_df[(self.ref_df['job_id'] == test_job_id) & (
                                     self.ref_df['conf_id'].isin(known_confs))]]).values

        ref_nodes = self.ref_df.values
        trainset = self.dataset.construct_trainset(raw_trainset=train_nodes)
        # print("Total unknowns SGD:", (31 * 16) - len(train_nodes))
        # print("Total TRAINSIZE SGD:", len(train_nodes))
        # model = SVD(n_factors=15, lr_all=.01, reg_bu=0, reg_bi=0, reg_pu=.5, reg_qi=0.05, n_epochs=n_epoch)
        if self.metric == "cost*perf":
            model = SVD(n_factors=15, lr_all=.001, reg_bu=0, reg_bi=0, reg_pu=5, reg_qi=5, n_epochs=10)
        elif self.metric == "cost":
            model = SVD(n_factors=15, lr_all=.001, reg_bu=0, reg_bi=0, reg_pu=5, reg_qi=5, n_epochs=10)
        else:
            model = SVD(n_factors=15, lr_all=.01, reg_bu=0, reg_bi=0, reg_pu=.05, reg_qi=5, n_epochs=5)
        start = time.time()
        model.fit(trainset)

        self.sgd_train_time += (time.time() - start)
        ref_list = list(filter(lambda x: x[0] == test_job_id, ref_nodes))
        test_set = self.dataset.construct_testset(raw_testset=ref_list)
        predictions = model.test(test_set)

        # rmse_perf = get_rmse(predictions)
        # print("RMSE using SGD perf:", rmse_perf)

        def apply_objective(x):
            config = self.config_info[int(x.iid)]

            est_real = self.denormalize(x.est)
            r_ui_real = self.denormalize(x.r_ui)
            if self.metric == "perf":
                return x
            elif self.metric == "cost":
                est = self.normalize((config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * est_real)
                r_ui = self.normalize(
                    (config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * r_ui_real)
                return Prediction(x.uid, x.iid, r_ui, est, x.details)
            else:
                est = self.normalize(
                    (config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * (est_real ** 2))
                r_ui = self.normalize(
                    (config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * (r_ui_real ** 2))
                return Prediction(x.uid, x.iid, r_ui, est, x.details)

        predictions = [apply_objective(x) for x in predictions]

        rmse_obj = get_rmse(predictions)
        # print("RMSE using SGD Obj:", rmse_obj)

        pred_best = min(predictions, key=lambda x: x.est)
        true_best = min(predictions, key=lambda x: x.r_ui)
        cost_diff = self.get_avg_difference(pred_best.r_ui, true_best.r_ui)
        # print("True conf_id SGD:", true_best.iid)
        return cost_diff, pred_best.iid, rmse_obj

    def evaluate_job(self, job_id, train_df, n_epochs):
        als_job_score = []
        sgd_job_score = []
        ref_confs_als = copy(self.ref_confs)
        ref_confs_sgd = copy(self.ref_confs)
        total_rmse_als = 0
        total_rmse_sgd = 0

        for epoch in range(n_epochs):
            err_als, next_conf_als, rmse_als = self.run_als(train_df, job_id, ref_confs_als, 10)
            err_sgd, next_conf_sgd, rmse_sgd = self.run_sgd(train_df, job_id, ref_confs_sgd, 10)
            # print("next_conf_als:", next_conf_als)
            # print("next_conf_sgd:", next_conf_sgd)
            if next_conf_sgd not in ref_confs_sgd:
                ref_confs_sgd.append(int(next_conf_sgd))
            if next_conf_als not in ref_confs_als:
                ref_confs_als.append(int(next_conf_als))
            als_job_score.append(err_als)
            sgd_job_score.append(err_sgd)
            total_rmse_als += rmse_als
            total_rmse_sgd += rmse_sgd
            self.total_trains += 1

        return als_job_score, sgd_job_score, total_rmse_als / self.n_epochs, total_rmse_sgd / self.n_epochs

    def run_evaluation(self, data_dir, res_dir, n_trials):
        als_res_df_list = [[] for _ in range(self.n_epochs)]
        sgd_res_df_list = [[] for _ in range(self.n_epochs)]
        res_subscript = f"{self.evolving_dataset}_{self.metric}"

        for trial in range(n_trials):
            print(f"TRIAL:\t {trial}")
            result_dim = (len(self.test_job_id_list), 1)
            eval_data_als = [np.full(result_dim, -1.0) for _ in range(self.n_epochs)]
            eval_data_sgd = [np.full(result_dim, -1.0) for _ in range(self.n_epochs)]
            train_df_list = []

            train_df_list.append(self.train_df)

            all_job_rmse_als = 0
            all_job_rmse_sgd = 0
            for index, job_id in tqdm(enumerate(self.test_job_id_list), total=len(self.test_job_id_list)):
                for den_id, train_df in enumerate(train_df_list):
                    als_job_scores, sgd_job_scores, als_job_rmse, sgd_job_rmse = self.evaluate_job(job_id, train_df,
                                                                                                   self.n_epochs)
                    all_job_rmse_als += als_job_rmse
                    all_job_rmse_sgd += sgd_job_rmse
                    for epoch, (als_score, sgd_score) in enumerate(zip(als_job_scores, sgd_job_scores)):
                        eval_data_als[epoch][index][den_id] = als_score
                        eval_data_sgd[epoch][index][den_id] = sgd_score
            # print(f"avg_rmse_als:{all_job_rmse_als / len(self.test_job_id_list)}\navg_rmse_sgd:{all_job_rmse_sgd / len(self.test_job_id_list)}")
            for i, (data_als, data_sgd) in enumerate(zip(eval_data_als, eval_data_sgd)):
                als_res_df = pd.DataFrame(data_als, columns=self.density_list, index=self.test_job_id_list)
                sgd_res_df = pd.DataFrame(data_sgd, columns=self.density_list, index=self.test_job_id_list)
                als_res_df_list[i].append(als_res_df)
                sgd_res_df_list[i].append(sgd_res_df)

        # print(f"AVG train time ALS:{self.als_train_time / self.total_trains}")
        # print(f"AVG train time SGD:{self.sgd_train_time / self.total_trains}")

        def get_mean_df(df_list):
            df_concat = pd.concat(df_list)
            by_row_index = df_concat.groupby(df_concat.index)
            df_means = by_row_index.mean()
            return df_means

        for i, (als_df_list, sgd_df_list) in enumerate(zip(als_res_df_list, sgd_res_df_list)):
            als_res_df = get_mean_df(als_df_list)
            sgd_res_df = get_mean_df(sgd_df_list)
            als_res_df.to_csv(f"{res_dir}/als_res_df_epoch_{i}_{res_subscript}.csv", index=False)
            sgd_res_df.to_csv(f"{res_dir}/sgd_res_df_epoch_{i}_{res_subscript}.csv", index=False)

    # def plot_result(self, res_dir):
    #
    #     line_styles = [":", "-"]  # "-.", "--",
    #     plt.figure(figsize=(15, 10))
    #     plt.rcParams.update({'font.size': 8})
    #     svd_best = [0, 0]
    #     res_subscript = f"{self.evolving_dataset}_{self.metric}"
    #
    #     for i in range(self.n_epochs):
    #         als_res_df = pd.read_csv(f"{res_dir}/als_res_df_epoch_{i}_{res_subscript}.csv")
    #         sgd_res_df = pd.read_csv(f"{res_dir}/sgd_res_df_epoch_{i}_{res_subscript}.csv")
    #         for ind, density in enumerate(self.density_list):
    #             als_prob = [sum(i <= thres for i in als_res_df[str(density)]) / len(als_res_df) for thres in
    #                         self.thres_list]
    #             sgd_prob = [sum(i <= thres for i in sgd_res_df[str(density)]) / len(sgd_res_df) for thres in
    #                         self.thres_list]
    #
    #             als_15p_prob = sum(i <= 0.15 for i in als_res_df[str(density)]) / len(als_res_df)
    #             sgd_15p_prob = sum(i <= 0.15 for i in sgd_res_df[str(density)]) / len(sgd_res_df)
    #             if als_15p_prob > svd_best[0]:
    #                 svd_best[0] = als_15p_prob
    #             if sgd_15p_prob > svd_best[1]:
    #                 svd_best[1] = sgd_15p_prob
    #
    #             plt.subplot(3, 3, ind + 1)
    #             plt.plot(self.thres_list, als_prob, line_styles[i % 4], label=f'ALS_epoch_{i}', color='green')
    #             plt.plot(self.thres_list, sgd_prob, line_styles[i % 4], label=f'SGD_epoch_{i}', color='purple')
    #             plt.grid(True)
    #             plt.xticks(self.thres_list)
    #             plt.yticks(self.thres_list)
    #
    #             plt.xlabel(f"row density {density * 100}%")
    #             plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #                        ncol=4, mode="expand", borderaxespad=0.1)
    #     plt.tight_layout(h_pad=1)
    #     plt.savefig(f"{res_dir}/fig_evaluation_{res_subscript}.png")
    #
    #     plt.clf()
    #     other_algo_perf = self.compare_algorithms()
    #     comparisons = svd_best + other_algo_perf
    #     # print(comparisons)
    #     with open(f"{res_dir}/comparison_{res_subscript}.txt", "w") as file:
    #         file.write(','.join(str(x) for x in comparisons))
    #         file.close()
    #     plt.bar(['ALS', 'SGD', 'Default', 'Max cost per time', 'Min cost per time'], comparisons)
    #     plt.savefig(f"{res_dir}/fig-comparison_{res_subscript}.png")

    def compare_algorithms(self):
        default_conf = None
        max_exp_conf = None
        min_exp_conf = None
        max_cost = 0
        min_cost = 2 ** 32
        for k, config in self.config_info.items():
            cost = config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate
            if config["cpus"] == 4 and config["memory"] == 256:
                default_conf = k
            if cost < min_cost:
                min_cost = cost
                min_exp_conf = k
            if cost > max_cost:
                max_cost = cost
                max_exp_conf = k

        max_value = 300102.0
        min_value = 1.3333333333333333

        def denormalize(value):
            value = (value * (max_value - min_value)) + min_value
            return value

        def calculate_cost(row):
            conf_id = row['conf_id']
            runtime = denormalize(row['cost'])
            config = self.config_info[conf_id]
            if self.metric == "perf":
                cost = self.normalize(runtime)
            elif self.metric == "cost":
                cost = self.normalize((config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * runtime)
            else:
                cost = self.normalize(
                    (config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * (runtime ** 2))
            return cost

        data = [0, 0, 0]
        total = 0
        for name, group in self.ref_df.groupby('job_id'):
            group['cost'] = group.apply(calculate_cost, axis=1)
            optimum_cost = group[group.cost == group.cost.min()]['cost'].values[0]
            default_conf_cost = group[group.conf_id == default_conf]['cost'].values[0]
            max_exp_conf_cost = group[group.conf_id == max_exp_conf]['cost'].values[0]
            min_exp_conf_cost = group[group.conf_id == min_exp_conf]['cost'].values[0]
            if self.get_avg_difference(default_conf_cost, optimum_cost) <= 0.15:
                data[0] += 1
            if self.get_avg_difference(max_exp_conf_cost, optimum_cost) <= 0.15:
                data[1] += 1
            if self.get_avg_difference(min_exp_conf_cost, optimum_cost) <= 0.15:
                data[2] += 1
            total += 1

        return [x / total for x in data]


def eval_evolving_ds(dataset):
    train_size = 99
    for metric in ["cost", "perf", "cost*perf"]:
        benchmark = Benchmark(metric=metric, train_size=train_size, dataset=dataset)
        # benchmark.debug("eval")
        # /home/himel/Documents/Academic/LOO_results/perf/57_jobs
        print(f"############### train_size:{train_size} metric: {metric} ###############")
        benchmark.run_evaluation("", "eval/evolving_ds", 2)
        # benchmark.plot_result(f"eval/evolving_ds")
        print(benchmark.compare_algorithms())


if __name__ == '__main__':
    # for train_size in [2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]:
    import argparse

    parser = argparse.ArgumentParser(description="Run Benchmark For Evolving Dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--train_dateset", default='medium', choices=['small', 'medium', 'large'],
                        help="size of training data subset")
    args = parser.parse_args()
    config = vars(args)
    train_dateset = config['train_dateset']
    start = time.time()
    eval_evolving_ds(train_dateset)
    print(f"Elapsed time: {(time.time() - start) / 60} min")
