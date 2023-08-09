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
    def __init__(self, metric, train_size, params):
        self.data_file = "./dataset/dataset_full.csv"
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
        self.ref_df = self.init_data()
        self.reader = Reader(rating_scale=(0, 1))
        self.dataset = Dataset(reader=self.reader)
        self.n_epochs = 2
        self.n_jobs = len(self.ref_df.job_id.unique())
        self.job_id_list = range(int(len(self.ref_df.job_id.unique())))
        # RMSE invalid if multiple density
        self.density_list = [1.0]
        self.train_size = train_size
        self.test_job_id_list = range(int(self.train_size))
        self.thres_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.als_train_time = 0
        self.sgd_train_time = 0
        self.total_trains = 0
        self.params = params
        # self.K = K
        # self.lamda_u = lamda_u
        # self.lamda_v = lamda_v
        # self.lr = lr
        # self.training_epochs = training_epochs

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

        train_jobs_ids = np.random.choice(self.job_id_list, self.train_size, replace=False)
        nodes_per_conf = [0] * (self.n_configs - len(self.ref_confs))
        weights = self.get_weights(nodes_per_conf, int(n_samples * self.train_size / self.n_configs))
        # print("=" * 10, density, "=" * 10)
        for name, group in self.ref_df.groupby('job_id'):
            if name not in train_jobs_ids:
                continue
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

    def df_to_mat(self, df):
        temp_df = df.drop("timestamp", axis=1)
        matrix = np.array(temp_df.pivot(index='job_id', columns='conf_id', values='cost').fillna(self.empty_val))
        return matrix

    def run_als(self, train_df, test_job_id, known_confs):
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

        model = ALS(X=train_X, K=self.params['K'],
                    lamb_u=self.params['lambda_u'], lamb_v=self.params['lambda_v'], none_val=self.empty_val,
                    max_epoch=self.params['training_epochs'])
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

    def run_sgd(self, train_df, test_job_id, known_confs):

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
                                 self.ref_df[(self.ref_df['job_id'] == test_job_id) & (
                                     self.ref_df['conf_id'].isin(known_confs))]]).values

        ref_nodes = self.ref_df.values
        trainset = self.dataset.construct_trainset(raw_trainset=train_nodes)

        model = SVD(n_factors=self.params['K'], lr_all=self.params['lr'],
                    reg_bu=0, reg_bi=0, reg_pu=self.params['lambda_u'], reg_qi=self.params['lambda_v'],
                    n_epochs=self.params['training_epochs'])
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

    def evaluate_job(self, job_id, train_df, n_epochs, alg):
        job_score = []
        ref_confs = copy(self.ref_confs)
        total_rmse = 0

        for epoch in range(n_epochs):
            if alg == "als":
                err, next_conf, rmse = self.run_als(train_df, job_id, ref_confs)

            else:
                err, next_conf, rmse = self.run_sgd(train_df, job_id, ref_confs)

            if next_conf in ref_confs:
                next_conf = random.choice(list(set([int(x) for x in self.config_info.keys()]) - set(ref_confs)))

            ref_confs.append(int(next_conf))
            job_score.append(err)
            total_rmse += rmse
            self.total_trains += 1

        return job_score, total_rmse / self.n_epochs

    def run_evaluation(self, data_dir, res_dir, n_trials, alg):
        res_df_list = [[] for _ in range(self.n_epochs)]
        res_subscript = f"{self.metric}"

        for trial in range(n_trials):
            print(f"TRIAL:\t {trial}")
            result_dim = (len(self.test_job_id_list), len(self.density_list))
            eval_data = [np.full(result_dim, -1.0) for _ in range(self.n_epochs)]
            train_df_list = []

            for density in self.density_list:
                train_df = self.create_train_df(density)
                train_df_list.append(train_df)

            all_job_rmse = 0
            for index, job_id in tqdm(enumerate(self.test_job_id_list), total=len(self.test_job_id_list)):
                for den_id, train_df in enumerate(train_df_list):
                    job_scores, job_rmse = self.evaluate_job(job_id, train_df, self.n_epochs, alg)
                    all_job_rmse += job_rmse
                    for epoch, score in enumerate(job_scores):
                        eval_data[epoch][index][den_id] = score
            # print(f"avg_rmse_{alg}:{all_job_rmse / len(self.test_job_id_list)}")
            for i, data in enumerate(eval_data):
                res_df = pd.DataFrame(data, columns=self.density_list, index=self.test_job_id_list)
                res_df_list[i].append(res_df)

        # print(f"AVG train time {alg.upper()}:{self.als_train_time / self.total_trains}")

        def get_mean_df(df_list):
            df_concat = pd.concat(df_list)
            by_row_index = df_concat.groupby(df_concat.index)
            df_means = by_row_index.mean()
            return df_means

        for i, df_list in enumerate(res_df_list):
            res_df = get_mean_df(df_list)
            res_df.to_csv(f"{res_dir}/{alg}_res_df_epoch_{i}_{res_subscript}.csv", index=False)

    def eval_result(self, res_dir, alg):

        line_styles = [":", "-"]  # "-.", "--",
        plt.figure(figsize=(15, 10))
        plt.rcParams.update({'font.size': 8})
        svd_best = 0
        res_subscript = f"{self.metric}"

        for i in range(self.n_epochs):
            res_df = pd.read_csv(f"{res_dir}/{alg}_res_df_epoch_{i}_{res_subscript}.csv")
            for ind, density in enumerate(self.density_list):
                prob = [sum(i <= thres for i in res_df[str(density)]) / len(res_df) for thres in self.thres_list]

                thres_15p_prob = sum(i <= 0.15 for i in res_df[str(density)]) / len(res_df)
                if thres_15p_prob > svd_best:
                    svd_best = thres_15p_prob

                plt.subplot(3, 3, ind + 1)
                plt.plot(self.thres_list, prob, line_styles[i % 4], label=f'ALS_epoch_{i}', color='green')
                plt.grid(True)
                plt.xticks(self.thres_list)
                plt.yticks(self.thres_list)

                plt.xlabel(f"row density {density * 100}%")
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                           ncol=4, mode="expand", borderaxespad=0.1)
        plt.tight_layout(h_pad=1)
        plt.savefig(f"{res_dir}/fig_evaluation_{res_subscript}.png")

        plt.clf()
        other_algo_perf = self.compare_algorithms()
        comparisons = [svd_best] + other_algo_perf
        # print(comparisons)
        with open(f"{res_dir}/comparison_{res_subscript}.txt", "w") as file:
            file.write(','.join(str(x) for x in comparisons))
            file.close()
        plt.bar([alg.upper(), 'Default', 'Max cost per time', 'Min cost per time'], comparisons)
        plt.savefig(f"{res_dir}/fig-comparison_{res_subscript}.png")

        plt.close()
        return svd_best

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


if __name__ == '__main__':
    alg = 'als'
    metric = 'cost*perf'
    params_grid = []
    for K in [10, 15, 30]:
        for lambda_u in [5, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005]:
            for lambda_v in [5, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]:
                for lr in [1, 0.1, 0.01, 0.001] if alg == 'sgd' else [1]:
                    for training_epochs in [5, 10, 15, 30, 50]:
                        params_grid.append({"K": K, "lambda_u": lambda_u,
                                            "lambda_v": lambda_v,
                                            "lr": lr, "training_epochs": training_epochs})

    print(len(params_grid))
    best_quota = 0.95
    for train_size in range(2, 100):
        param_data = {}
        for params in params_grid:
            benchmark = Benchmark(metric=metric, train_size=train_size, params=params)
            benchmark.run_evaluation("", "eval", 1, alg)
            svd_score = benchmark.eval_result("eval", alg)
            # svd_score = random.uniform(0, 1)
            param_data[json.dumps(params)] = svd_score
        top_n = max(1, int(len(params_grid) * best_quota))
        # params_grid = params_grid[:top_n]
        # print(train_size, ':', len(params_grid))
        params_grid = [json.loads(params) for params, score in sorted(param_data.items(),
                                                                      key=lambda item: item[1],
                                                                      reverse=True)[:top_n]]
        if len(params_grid) == 1:
            break

    print(">>>", metric, alg, "best_params:", params_grid)
    # train_size = 90
    #
    # metrics = ["cost", "cost*perf", "perf"]
    # for metric in metrics:
    #     for alg in ["als", "sgd"]:
    #         best_score = 0
    #         best_params = {}
    #         for K in [10, 15, 30, 50]:
    #             for lambda_u in [5, 0.5, 0.05, 0.005]:
    #                 for lambda_v in [5, 0.5, 0.05, 0.005, 0.0005]:
    #                     lr_list = [1, 0.1, 0.01, 0.001] if alg == "sgd" else [1]
    #                     for lr in lr_list:
    #                         for training_epochs in [5, 10, 30, 50, 100]:
    #                             print({"K": K, "lambda_u": lambda_u,
    #                                    "lambda_v": lambda_v,
    #                                    "lr": lr, "training_epochs": training_epochs})
    #                             benchmark = Benchmark(metric=metric, train_size=train_size, K=K, lamda_u=lambda_u,
    #                                                   lamda_v=lambda_v, lr=lr, training_epochs=training_epochs)
    #                             benchmark.run_evaluation("", "eval", 5, alg)
    #                             svd_score = benchmark.eval_result("eval", alg)
    #                             if svd_score - best_score > 0.01:
    #                                 best_score = svd_score
    #                                 best_params = {"K": K, "lambda_u": lambda_u,
    #                                                "lambda_v": lambda_v,
    #                                                "lr": lr, "training_epochs": training_epochs}
    #
    #         print('=' * 50)
    #         print(">>>", metric, alg, best_score, best_params)
    #         print('=' * 50)

    # benchmark = Benchmark(metric=metric, train_size=train_size, K=15,
    #                       lamda_u=0.05, lamda_v=0.05, lr=1,
    #                       training_epochs=30)
    # benchmark.run_evaluation("", "eval", 1, "als")
    # svd_score = benchmark.eval_result("eval", "als")
    # print(svd_score)

