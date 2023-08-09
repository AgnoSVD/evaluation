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
    def __init__(self, metric, train_size, density_list, als_params, sgd_params):
        self.data_file = "./dataset/dataset_full.csv"
        self.worker_conf = "worker_conf.json"
        self.config_info = json.load(open(self.worker_conf))
        self.config_info = dict(zip([str(int(x) - 1).zfill(2) for x in self.config_info.keys()], self.config_info.values()))
        self.n_configs = len(self.config_info)
        self.cpu_rate = 0.0000001
        self.memory_rate = 0.000000001
        self.empty_val = 99
        self.ref_confs = [21, 6]  # [27, 7]
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
        self.ref_df = self.init_data()
        self.reader = Reader(rating_scale=(0, 1))
        self.dataset = Dataset(reader=self.reader)
        self.n_epochs = 2
        self.n_jobs = len(self.ref_df.job_id.unique())
        self.job_id_list = range(int(len(self.ref_df.job_id.unique())))
        # RMSE invalid if multiple density
        self.density_list = density_list
        self.train_size = train_size
        self.test_job_id_list = range(int(self.train_size))
        self.thres_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.als_train_time = 0
        self.sgd_train_time = 0
        self.total_trains = 0
        self.als_params = als_params
        self.sgd_params = sgd_params

    def normalize(self, value):
        value = (value - self.min_value) / (self.max_value - self.min_value)
        return value

    def denormalize(self, value):
        value = (value * (self.max_value - self.min_value)) + self.min_value
        return value

    def init_data(self):
        data_df = pd.read_csv(self.data_file, header=None, names=['job_id', 'conf_id', 'cost', "timestamp"])
        data_df = data_df[data_df['conf_id'].isin([int(x) for x in self.config_info.keys()])]
        if data_df.job_id.min() != 0:  # for re indexing from 0 when eval only sb
            data_df.job_id = data_df.job_id - 57
        max_value = data_df['cost'].max()
        min_value = data_df['cost'].min()

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
        self.ref_confs = [conf_rename[x] for x in self.ref_confs]
        self.config_info = conf_info
        return data_df

    def get_weights(self, nodes_per_conf, n_samples):
        weights = [max(n_samples - x, 1e-10) / n_samples for x in nodes_per_conf]
        weights = [x / sum(weights) for x in weights]  # np.exp(weights - np.max(weights))
        return weights

    def update_nodes_per_conf(self, nodes_per_conf, sampled_rows):
        for conf_id in sampled_rows.index:
            nodes_per_conf[conf_id] += 1
        return nodes_per_conf

    def sample_np(self, non_ref_rows, weights, seed, n_samples):
        np.random.seed(seed)
        non_ref_rows = non_ref_rows.sort_values(by=['conf_id'], ascending=True).reset_index(drop=True)
        sampled_indices = np.random.choice(non_ref_rows.index, n_samples, p=weights, replace=False)
        sampled_rows = non_ref_rows[non_ref_rows.index.isin(sampled_indices)]
        return sampled_rows

    def create_train_df(self, density):
        # print(f"evaluating with sparsity: {sparsity * 100}%")
        train_nodes = []
        total_known = 0
        seed = 100
        n_samples = int(density * self.n_configs) - len(self.ref_confs)
        np.random.seed(2281)
        # try to use rows corresponding to the same dataset scale factor
        if self.train_size % 3 == 0:
            train_jobs_ids = []
            for job_id in np.random.choice(range(0, self.n_jobs, 3), int(self.train_size / 3), replace=False):
                train_jobs_ids += [job_id, job_id + 1, job_id + 2]
        else:
            train_jobs_ids = np.random.choice(self.job_id_list, self.train_size, replace=False)
        nodes_per_conf = [0] * (self.n_configs - len(self.ref_confs))
        weights = self.get_weights(nodes_per_conf, int(n_samples * self.train_size / self.n_configs))
        # print("=" * 10, density, "=" * 10)
        for name, group in self.ref_df.groupby('job_id'):
            if name not in train_jobs_ids:
                continue
            if len(group)-len(self.ref_confs) == n_samples:
                subgroup = group
            else:
                ref_conf_rows = group[group['conf_id'].isin(self.ref_confs)]
                non_ref_rows = group[~group['conf_id'].isin(self.ref_confs)]
                sampled_rows = self.sample_np(non_ref_rows, weights, seed, n_samples)
                self.update_nodes_per_conf(nodes_per_conf, sampled_rows)
                weights = self.get_weights(nodes_per_conf, int(n_samples * self.train_size / self.n_configs))
                subgroup = pd.concat([ref_conf_rows, sampled_rows])
            total_known += len(subgroup)
            for index, row in subgroup.iterrows():
                train_nodes.append(row.values)
            seed += 10

        train_df = pd.DataFrame(np.array(train_nodes), columns=['job_id', 'conf_id', 'cost', 'timestamp'])
        # print(train_df.groupby('conf_id')['conf_id'].count().reset_index(name="count")['count'].values)
        return train_df

    def get_avg_difference(self, v_2, v_1):
        v_2 = self.denormalize(v_2)
        v_1 = self.denormalize(v_1)
        value = abs(v_2 - v_1) / ((v_2 + v_1) / 2)
        return value

    def get_improvement(self, v_new, v_default):
        v_new = self.denormalize(v_new)
        v_default = self.denormalize(v_default)
        return 1-(v_new/v_default)

    def df_to_mat(self, df):
        temp_df = df.drop("timestamp", axis=1)
        matrix = np.array(temp_df.pivot(index='job_id', columns='conf_id', values='cost').fillna(self.empty_val))
        return matrix

    def run_als(self, train_df, test_job_id, known_confs, K, n_epochs):
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
        train_X[test_job_id] = np.full(self.n_configs, self.empty_val)
        train_X[test_job_id][known_confs] = ref_X[test_job_id][known_confs]
        model = ALS(X=train_X, K=self.als_params[self.metric]['K'],
                    lamb_u=self.als_params[self.metric]['lambda_u'],
                    lamb_v=self.als_params[self.metric]['lambda_v'],
                    none_val=self.empty_val,
                    max_epoch=self.als_params[self.metric]['max_epochs'])
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

    def run_sgd(self, train_df, test_job_id, known_confs, K, n_epochs):

        def get_rmse(predictions):
            denormalized = []
            for x in predictions:
                if x.iid not in self.ref_confs:
                    r_ui = self.denormalize(x.r_ui)
                    est = self.denormalize(x.est)
                    denormalized.append(Prediction(x.uid, x.iid, r_ui, est, x.details))
            rmse = accuracy.rmse(denormalized, verbose=False)
            return rmse

        test_job_id = train_df.job_id.unique()[test_job_id]  # test_job_id is an index
        train_nodes = pd.concat([train_df[train_df['job_id'] != test_job_id],
                                 self.ref_df[(self.ref_df['job_id'] == test_job_id) & (self.ref_df['conf_id'].isin(known_confs))]]).values

        ref_nodes = self.ref_df.values
        trainset = self.dataset.construct_trainset(raw_trainset=train_nodes)
        model = SVD(n_factors=self.sgd_params[self.metric]['K'],
                    lr_all=self.sgd_params[self.metric]['lr'],
                    reg_bu=0, reg_bi=0,
                    reg_pu=self.sgd_params[self.metric]['lambda_u'],
                    reg_qi=self.sgd_params[self.metric]['lambda_v'],
                    n_epochs=self.sgd_params[self.metric]['max_epochs'])
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
                r_ui = self.normalize((config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * r_ui_real)
                return Prediction(x.uid, x.iid, r_ui, est, x.details)
            else:
                est = self.normalize((config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * (est_real ** 2))
                r_ui = self.normalize((config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * (r_ui_real ** 2))
                return Prediction(x.uid, x.iid, r_ui, est, x.details)

        predictions = [apply_objective(x) for x in predictions]

        rmse_obj = get_rmse(predictions)
        # print("RMSE using SGD Obj:", rmse_obj)

        pred_best = min(predictions, key=lambda x: x.est)
        true_best = min(predictions, key=lambda x: x.r_ui)
        cost_diff = self.get_avg_difference(pred_best.r_ui, true_best.r_ui)
        # print("True conf_id SGD:", true_best.iid)
        return cost_diff, pred_best.iid, rmse_obj

    def evaluate_job(self, job_id, train_df, n_epochs, density):
        als_job_score = []
        sgd_job_score = []
        ref_confs_als = copy(self.ref_confs)
        ref_confs_sgd = copy(self.ref_confs)
        total_rmse_als = 0
        total_rmse_sgd = 0
        K_als = 15
        n_epochs_als = 10
        K_sgd = 15
        n_epochs_sgd = 10
        for epoch in range(n_epochs):
            err_als, next_conf_als, rmse_als = self.run_als(train_df, job_id, ref_confs_als, K_als, n_epochs_als)
            err_sgd, next_conf_sgd, rmse_sgd = self.run_sgd(train_df, job_id, ref_confs_sgd, K_sgd, n_epochs_sgd)
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

    def run_evaluation(self, res_dir, n_trials, n_best=None):
        als_res_df_list = [[] for _ in range(self.n_epochs)]
        sgd_res_df_list = [[] for _ in range(self.n_epochs)]
        res_subscript = f"{self.train_size}_{self.metric}"

        for trial in range(n_trials):
            print(f"TRIAL:\t {trial}")
            result_dim = (len(self.test_job_id_list), len(self.density_list))
            eval_data_als = [np.full(result_dim, -1.0) for _ in range(self.n_epochs)]
            eval_data_sgd = [np.full(result_dim, -1.0) for _ in range(self.n_epochs)]
            train_df_list = []

            for density in self.density_list:
                train_df = self.create_train_df(density)
                train_df_list.append(train_df)

            all_job_rmse_als = 0
            all_job_rmse_sgd = 0
            for index, job_id in tqdm(enumerate(self.test_job_id_list), total=len(self.test_job_id_list)):
                for den_id, train_df in enumerate(train_df_list):
                    (als_job_scores, sgd_job_scores,
                     als_job_rmse, sgd_job_rmse) = self.evaluate_job(job_id, train_df,
                                                                     self.n_epochs, self.density_list[den_id])
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
            # df_concat['sum'] = df_concat.sum(axis=1)
            df_concat['sum'] = df_concat[df_concat < 0.15].count(axis=1)
            df_concat = df_concat.sort_values(by=['sum'])
            by_row_index = df_concat.groupby(df_concat.index)
            if n_best is None:
                df_means = by_row_index.mean()
            else:
                assert n_best <= n_trials, "must be n_best <= n_trials"
                by_row_index = by_row_index.head(n_best)
                by_row_index = by_row_index.groupby(by_row_index.index)
                df_means = by_row_index.min()
            df_means.drop('sum', axis=1, inplace=True)
            return df_means

        for i, (als_df_list, sgd_df_list) in enumerate(zip(als_res_df_list, sgd_res_df_list)):
            als_res_df = get_mean_df(als_df_list)
            sgd_res_df = get_mean_df(sgd_df_list)
            als_res_df.to_csv(f"{res_dir}/als_res_df_epoch_{i}_{res_subscript}.csv", index=False)
            sgd_res_df.to_csv(f"{res_dir}/sgd_res_df_epoch_{i}_{res_subscript}.csv", index=False)

            # als_impv_df.to_csv(f"{res_dir}/als_res_df_epoch_{i}_{res_subscript}.csv", index=False)
            # sgd_impv_df.to_csv(f"{res_dir}/sgd_res_df_epoch_{i}_{res_subscript}.csv", index=False)

    def log_comparison(self, res_dir):

        # line_styles = [":", "-"]  # "-.", "--",
        # plt.figure(figsize=(15, 10))
        # plt.rcParams.update({'font.size': 8})
        svd_best = [0, 0]
        res_subscript = f"{self.train_size}_{self.metric}"

        for i in range(self.n_epochs):
            als_res_df = pd.read_csv(f"{res_dir}/als_res_df_epoch_{i}_{res_subscript}.csv")
            sgd_res_df = pd.read_csv(f"{res_dir}/sgd_res_df_epoch_{i}_{res_subscript}.csv")
            for ind, density in enumerate(self.density_list):
                # als_prob = [sum(i <= thres for i in als_res_df[str(density)]) / len(als_res_df) for thres in self.thres_list]
                # sgd_prob = [sum(i <= thres for i in sgd_res_df[str(density)]) / len(sgd_res_df) for thres in self.thres_list]

                als_15p_prob = sum(i <= 0.15 for i in als_res_df[str(density)]) / len(als_res_df)
                sgd_15p_prob = sum(i <= 0.15 for i in sgd_res_df[str(density)]) / len(sgd_res_df)
                if als_15p_prob > svd_best[0]:
                    svd_best[0] = als_15p_prob
                if sgd_15p_prob > svd_best[1]:
                    svd_best[1] = sgd_15p_prob

                # plt.subplot(3, 3, ind + 1)
                # plt.plot(self.thres_list, als_prob, line_styles[i % 4], label=f'ALS_epoch_{i}', color='green')
                # plt.plot(self.thres_list, sgd_prob, line_styles[i % 4], label=f'SGD_epoch_{i}', color='purple')
                # plt.grid(True)
                # plt.xticks(self.thres_list)
                # plt.yticks(self.thres_list)

                # plt.xlabel(f"row density {density * 100}%")
                # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                #            ncol=4, mode="expand", borderaxespad=0.1)
        # plt.tight_layout(h_pad=1)
        # plt.savefig(f"{res_dir}/fig_evaluation_{res_subscript}.png")

        # plt.clf()
        other_algo_perf = self.compare_algorithms()
        comparisons = svd_best + other_algo_perf
        # print(comparisons)
        with open(f"{res_dir}/comparison_{res_subscript}.txt", "w") as file:
            file.write(','.join(str(x) for x in comparisons))
            file.close()
        # plt.bar(['ALS', 'SGD', 'Default', 'Sizeless', 'Max cost per time', 'Min cost per time'], comparisons)
        # plt.savefig(f"{res_dir}/fig-comparison_{res_subscript}.png")

    def compare_algorithms(self):
        default_conf = None
        max_exp_conf = None
        min_exp_conf = None
        max_cost = 0
        min_cost = 2 ** 32
        # collected from sizeless eval in <url>
        sizeless_probs = {
            "perf": 0.6818181818181817,
            "cost": 0.8181818181818183,
            "cost*perf": 0.7727272727272727
        }
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
                cost = self.normalize((config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * (runtime ** 2))
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

        return [sizeless_probs[self.metric]] + [x / total for x in data]


def eval_accuracy():
    als_params = {
        "cost*perf":    {"K": 10, "lambda_u": 0.05, "lambda_v": 0.05,  "max_epochs": 30},
        "cost":         {"K": 30, "lambda_u": 5, "lambda_v": 0.0005, "max_epochs": 10},
        "perf":         {"K": 50, "lambda_u": 0.5, "lambda_v": 0.0005, "max_epochs": 5}
    }
    sgd_params = {
        "cost*perf":    {"K": 15, "lambda_u": 5, "lambda_v": 5, "lr": 0.001, "max_epochs": 10},
        "cost":         {"K": 15, "lambda_u": 5, "lambda_v": 5, "lr": 0.01, "max_epochs": 10},
        "perf":         {"K": 15, "lambda_u": 0.05, "lambda_v": 5, "lr": 0.01, "max_epochs": 5}
    }
    train_size = 99
    density_list = [1.0]
    for metric in ["cost", "perf", "cost*perf"]:
        benchmark = Benchmark(metric=metric, train_size=train_size, density_list=density_list,
                              als_params=als_params, sgd_params=sgd_params)
        print(f"############### train_size:{train_size} metric: {metric} ###############")
        benchmark.run_evaluation("eval/default", 5)
        benchmark.log_comparison(f"eval/default")


def eval_steady_state():
    als_params = {
        # "cost":         {"K": 30, "lambda_u": 5, "lambda_v": 0.0005, "max_epochs": 10},
        "cost*perf":    {"K": 10, "lambda_u": 0.05, "lambda_v": 0.05,  "max_epochs": 30},
        "cost":         {"K": 10, "lambda_u": 0.05, "lambda_v": 0.05, "max_epochs": 30},
        "perf":         {"K": 10, "lambda_u": 0.05, "lambda_v": 0.05, "max_epochs": 30}
    }
    sgd_params = {
        "cost*perf":    {"K": 15, "lambda_u": 5, "lambda_v": 5, "lr": 0.001, "max_epochs": 10},
        "cost":         {"K": 15, "lambda_u": 5, "lambda_v": 5, "lr": 0.001, "max_epochs": 5},
        "perf":         {"K": 15, "lambda_u": 0.05, "lambda_v": 5, "lr": 0.01, "max_epochs": 5}
    }
    # density_list = np.linspace(0.2, 1, 5)
    # density_list = np.linspace(0.2, 1, 9)
    # density_list = np.linspace(0.2, 1, 41)
    density_list = np.linspace(0.2, 1, 81)
    train_size = 99
    for metric in ["cost", "perf", "cost*perf"]:
        benchmark = Benchmark(metric=metric, train_size=train_size, density_list=density_list,
                              als_params=als_params, sgd_params=sgd_params)
        print(f"############### train_size:{train_size} metric: {metric} ###############")
        benchmark.run_evaluation("eval/row_density", 1)
        # benchmark.plot_result(f"eval/row_density")


def eval_cold_start():
    als_params = {
        # "cost*perf":    {"K": 10, "lambda_u": 0.05, "lambda_v": 0.05,  "max_epochs": 30},
        # "cost*perf":    {"K": 15, "lambda_u": 0.05, "lambda_v": 0.5, "max_epochs": 5},  # old
        # "cost":         {"K": 15, "lambda_u": 5, "lambda_v": 0.0005, "max_epochs": 10},  # old
        # "perf":         {"K": 30, "lambda_u": 0.05, "lambda_v": 0.0005, "max_epochs": 5}
        # "perf":         {"K": 50, "lambda_u": 0.5, "lambda_v": 0.0005, "max_epochs": 5}
        "cost*perf":    {"K": 15, "lambda_u": 0.01, "lambda_v": 0.5, "max_epochs": 11},
        "cost":         {"K": 15, "lambda_u": 0.01, "lambda_v": 0.5, "max_epochs": 15},
        "perf":         {"K": 15, "lambda_u": 0.01, "lambda_v": 0.5, "max_epochs": 15}
    }
    sgd_params = {
        "cost*perf":    {"K": 15, "lambda_u": 5, "lambda_v": 5, "lr": 0.001, "max_epochs": 10},
        "cost":         {"K": 15, "lambda_u": 5, "lambda_v": 5, "lr": 0.01, "max_epochs": 5},
        "perf":         {"K": 15, "lambda_u": 0.05, "lambda_v": 5, "lr": 0.01, "max_epochs": 5}
    }
    # density_list = np.linspace(0.2, 1, 5)
    # density_list = np.linspace(0.2, 1, 9)
    # density_list = np.linspace(0.2, 1, 41)
    density_list = np.linspace(0.2, 1, 81)
    train_size = 40
    for metric in ["cost", "perf", "cost*perf"]:
        benchmark = Benchmark(metric=metric, train_size=train_size, density_list=density_list,
                              als_params=als_params, sgd_params=sgd_params)
        print(f"############### train_size:{train_size} metric: {metric} ###############")
        benchmark.run_evaluation("eval/row_density", 1)
        # benchmark.plot_result(f"eval/row_density")


def eval_n_jobs():
    als_params = {
        # "cost*perf":    {"K": 15, "lambda_u": 0.05, "lambda_v": 0.5,  "max_epochs": 5},
        "cost*perf":    {'K': 15, 'lambda_u': 0.05, 'lambda_v': 0.005, 'max_epochs': 30},
        "cost":         {"K": 30, "lambda_u": 5, "lambda_v": 0.0005, "max_epochs": 10},
        "perf":         {"K": 50, "lambda_u": 0.5, "lambda_v": 0.0005, "max_epochs": 5}
    }
    sgd_params = {
        "cost*perf":    {"K": 15, "lambda_u": 5, "lambda_v": 5, "lr": 0.001, "max_epochs": 10},
        "cost":         {"K": 15, "lambda_u": 5, "lambda_v": 5, "lr": 0.01, "max_epochs": 10},
        "perf":         {"K": 15, "lambda_u": 0.05, "lambda_v": 5, "lr": 0.01, "max_epochs": 5}
    }
    density_list = [1.0]
    # for train_size in [2, 4, 8, 12, 16, 24, 33, 40, 48, 57, 66, 78, 82, 90, 99]:
    # for train_size in [12, 24, 40, 48, 78, 99]:
    for train_size in range(2, 100):
        for metric in ["cost", "perf", "cost*perf"]:
            benchmark = Benchmark(metric=metric, train_size=train_size, density_list=density_list,
                                  als_params=als_params, sgd_params=sgd_params)
            print(f"############### train_size:{train_size} metric: {metric} ###############")
            benchmark.run_evaluation("eval/n_jobs", 1)
            benchmark.log_comparison(f"eval/n_jobs")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run Benchmark",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--scenario", default="accuracy", choices=['accuracy', 'steady_state', 'cold_start', 'n_jobs'],
                        help="benchmark scenario")
    args = parser.parse_args()
    config = vars(args)
    start = time.time()
    scenario = config['scenario']
    if scenario == "accuracy":
        eval_accuracy()
    if scenario == "steady_state":
        eval_steady_state()
    if scenario == "cold_start":
        eval_cold_start()
    if scenario == "n_jobs":
        eval_n_jobs()
    print(f"Elapsed time: {(time.time() - start) / 60} min")
