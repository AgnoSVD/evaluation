import json
import random
import time
import warnings
from copy import copy
from math import floor

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, Prediction, accuracy, SVDpp, KNNBaseline
from tqdm import tqdm

from als_recommendation import ALS

import matplotlib.pyplot as plt


class Benchmark:
    def __init__(self, train_size, cost_restriction):
        self.data_file = "./dataset/dataset_full.csv"
        self.worker_conf = "worker_conf.json"
        self.config_info = json.load(open(self.worker_conf))
        self.config_info = dict(zip([str(int(x) - 1).zfill(2) for x in self.config_info.keys()], self.config_info.values()))
        self.n_configs = len(self.config_info)
        self.cpu_rate = 0.0000001
        self.memory_rate = 0.000000001
        self.empty_val = 99
        self.ref_confs = [21, 6]
        self.min_value = None
        self.max_value = None
        self.max_cost = 4.9762558480000007
        self.min_cost = 7.12e-07
        self.cost_restriction = cost_restriction
        self.ref_df = self.init_data()
        self.reader = Reader(rating_scale=(0, 1))
        self.dataset = Dataset(reader=self.reader)
        self.n_epochs = 2
        self.n_jobs = len(self.ref_df.job_id.unique())
        self.job_id_list = range(int(len(self.ref_df.job_id.unique())))
        self.density_list = [1.0]
        self.train_size = train_size
        self.evolving_dataset = False
        self.test_job_id_list = range(self.train_size)  # range(int(self.train_size / 3) * 0, int(self.train_size / 3) * 3)  # test only small dataset
        self.thres_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def normalize(self, value):
        value = (value - self.min_value) / (self.max_value - self.min_value)
        return value

    def denormalize(self, value):
        value = (value * (self.max_value - self.min_value)) + self.min_value
        return value

    def init_data(self):
        data_df = pd.read_csv(self.data_file, header=None, names=['job_id', 'conf_id', 'cost', "timestamp"])
        data_df = data_df[data_df['conf_id'].isin([int(x) for x in self.config_info.keys()])]

        self.min_value = data_df['cost'].min()
        self.max_value = data_df['cost'].max()
        # print(self.max_value)
        # print(self.min_value)
        data_df['cost'] = data_df['cost'].apply(self.normalize)

        conf_rename = {}
        conf_info = {}
        for index, conf_id in enumerate(self.config_info):
            conf_rename[int(conf_id)] = index
            conf_info[index] = self.config_info[conf_id]

        data_df.replace({'conf_id': conf_rename}, inplace=True)
        self.ref_confs = [conf_rename[x] for x in self.ref_confs]
        self.config_info = conf_info

        return data_df

    # def get_weights(self, nodes_per_conf, n_samples):
    #     weights = [max(n_samples - x, 1e-10) / n_samples for x in nodes_per_conf]
    #     weights = [x / sum(weights) for x in weights]  # np.exp(weights - np.max(weights))
    #     return weights
    #
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
    #
    # def sample_pd(self, non_ref_rows, weights, seed, n_samples):
    #     non_ref_rows = non_ref_rows.sort_values(by=['conf_id'], ascending=True)
    #     sampled_rows = non_ref_rows.sample(n=n_samples, weights=weights, random_state=seed, replace=False)
    #     for val in sampled_rows.conf_id.values:
    #         weights[val] /= 4
    #     weights = dict(zip(weights.keys(), [x / sum(weights.values()) for x in weights.values()]))
    #     return sampled_rows, weights

    def create_train_df(self):
        # print(f"evaluating with sparsity: {sparsity * 100}%")
        train_nodes = []
        total_known = 0
        # seed = None
        # n_samples = int(density * self.n_configs) - len(self.ref_confs)
        np.random.seed(10)
        train_jobs_ids = np.random.choice(self.job_id_list, self.train_size, replace=False)
        nodes_per_conf = [0] * (self.n_configs - len(self.ref_confs))
        # weights = self.get_weights(nodes_per_conf, int(n_samples * self.train_size / self.n_configs))
        # print("=" * 10, density, "=" * 10)
        for name, group in self.ref_df.groupby('job_id'):
            if name not in train_jobs_ids:
                continue
            # ref_conf_rows = group[group['conf_id'].isin(self.ref_confs)]
            # non_ref_rows = group[~group['conf_id'].isin(self.ref_confs)]
            # sampled_rows = self.sample_np(non_ref_rows, weights, seed, n_samples)
            # self.update_nodes_per_conf(nodes_per_conf, sampled_rows)
            # weights = self.get_weights(nodes_per_conf, int(n_samples * self.train_size / self.n_configs))
            # subgroup = pd.concat([ref_conf_rows, sampled_rows])
            subgroup = group
            total_known += len(subgroup)
            for index, row in subgroup.iterrows():
                train_nodes.append(row.values)
            # seed += 10

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
        train_X = self.df_to_mat(train_df)
        ref_X = self.df_to_mat(self.ref_df)
        if self.evolving_dataset:
            temp = ref_X[test_job_id][known_confs]
            test_job_id = train_X.shape[0]
            train_X = np.insert(train_X, test_job_id, np.full(self.n_configs, self.empty_val), axis=0)
            train_X[test_job_id][known_confs] = temp
        else:
            train_X[test_job_id] = np.full(self.n_configs, self.empty_val)
            train_X[test_job_id][known_confs] = ref_X[test_job_id][known_confs]
        # print("Total TRAINSIZE ALS:", len((np.where(train_X != self.empty_val)[0])))
        # model = ALS(X=train_X, K=5, lamb_u=.01, lamb_v=.001, none_val=self.empty_val, max_epoch=100)
        model = ALS(X=train_X, K=50, lamb_u=.5, lamb_v=.0005, none_val=self.empty_val, max_epoch=5)
        model.train()
        est_matrix = model.get_est_matrix()

        def apply_cost_restriction(runtime_list):
            """Basically change the runtime values to one to discard these,
            by taking minimum of the rest we get the best under restriction"""
            cost_list = []
            for conf_id, runtime in enumerate(runtime_list):
                runtime_real = self.denormalize(runtime)
                config = self.config_info[int(conf_id)]
                cost = (config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * runtime_real
                if (cost - self.min_cost) / (self.max_cost - self.min_cost) > self.cost_restriction:
                    runtime = 1.0
                cost_list.append(runtime)
            return np.array(cost_list)

        pred_list = apply_cost_restriction(est_matrix[test_job_id])
        true_list = apply_cost_restriction(ref_X[test_job_id])

        pred_conf_id = np.argmin(pred_list, axis=0)
        true_conf_id = np.argmin(true_list, axis=0)

        cost_diff = self.get_avg_difference(ref_X[test_job_id][pred_conf_id],
                                            ref_X[test_job_id][true_conf_id])
        return cost_diff, pred_conf_id

    def run_sgd(self, train_df, test_job_id, known_confs):
        if self.evolving_dataset:
            test_job_id = test_job_id  # test_job_id is not an index
        else:
            test_job_id = train_df.job_id.unique()[test_job_id]  # test_job_id is an index
        train_nodes = pd.concat([train_df[train_df['job_id'] != test_job_id],
                                 self.ref_df[(self.ref_df['job_id'] == test_job_id) & (self.ref_df['conf_id'].isin(known_confs))]]).values
        # print("Total TRAINSIZE SGD:", len(train_nodes))
        ref_nodes = self.ref_df.values
        trainset = self.dataset.construct_trainset(raw_trainset=train_nodes)
        model = SVD(n_factors=15, lr_all=.01, reg_bu=0, reg_bi=0, reg_pu=.05, reg_qi=5, n_epochs=5)
        model.fit(trainset)
        ref_list = list(filter(lambda x: x[0] == test_job_id, ref_nodes))
        test_set = self.dataset.construct_testset(raw_testset=ref_list)
        predictions = model.test(test_set)

        def apply_cost_restriction(x, type):
            """Basically change the runtime values to one to discard these,
            by taking minimum of the rest we get the best under restriction"""
            config = self.config_info[int(x.iid)]
            if type == "pred":
                runtime = x.est
            else:
                runtime = x.r_ui
            runtime_real = self.denormalize(runtime)
            cost = (config["cpus"] * self.cpu_rate + config["memory"] * self.memory_rate) * runtime_real
            if (cost - self.min_cost) / (self.max_cost - self.min_cost) > self.cost_restriction:
                runtime = 1.0
            return runtime


        pred_best = min(predictions, key=lambda x: apply_cost_restriction(x, "pred"))
        true_best = min(predictions, key=lambda x: apply_cost_restriction(x, "true"))

        cost_diff = self.get_avg_difference(pred_best.r_ui, true_best.r_ui)
        # print("True conf_id SGD:", true_best.iid)
        return cost_diff, pred_best.iid

    def evaluate_job(self, job_id, train_df, n_epochs):
        als_job_score = []
        sgd_job_score = []
        ref_confs_als = copy(self.ref_confs)
        ref_confs_sgd = copy(self.ref_confs)
        for epoch in range(n_epochs):
            err_als, next_conf_als = self.run_als(train_df, job_id, ref_confs_als)
            err_sgd, next_conf_sgd = self.run_sgd(train_df, job_id, ref_confs_sgd)
            als_job_score.append(err_als)
            sgd_job_score.append(err_sgd)
            if next_conf_sgd in ref_confs_sgd:
                continue
            if next_conf_als in ref_confs_als:
                continue
            ref_confs_als.append(int(next_conf_als))
            ref_confs_sgd.append(int(next_conf_sgd))

        return als_job_score, sgd_job_score

    def run_evaluation(self, data_dir, res_dir, n_trials):
        als_res_df_list = [[] for _ in range(self.n_epochs)]
        sgd_res_df_list = [[] for _ in range(self.n_epochs)]
        res_subscript = self.cost_restriction

        for trial in range(n_trials):
            print(f"TRIAL:\t {trial}")
            result_dim = (len(self.test_job_id_list), 1)
            eval_data_als = [np.full(result_dim, -1.0) for _ in range(self.n_epochs)]
            eval_data_sgd = [np.full(result_dim, -1.0) for _ in range(self.n_epochs)]
            train_df_list = []
            #
            # for density in self.density_list:
            train_df = self.create_train_df()
            if self.evolving_dataset:
                # now self.test_job_id_list will have real job ids
                train_job_ids = train_df.job_id.unique()
                test_job_id_list = set(train_job_ids) & set(self.test_job_id_list)
                train_df = train_df[~train_df['job_id'].isin(test_job_id_list)]
                print(self.df_to_mat(train_df).shape)
            train_df_list.append(train_df)

            for index, job_id in tqdm(enumerate(self.test_job_id_list), total=len(self.test_job_id_list)):
                for den_id, train_df in enumerate(train_df_list):
                    als_job_scores, sgd_job_scores = self.evaluate_job(job_id, train_df, self.n_epochs)
                    for epoch, (als_score, sgd_score) in enumerate(zip(als_job_scores, sgd_job_scores)):
                        eval_data_als[epoch][index][den_id] = als_score
                        eval_data_sgd[epoch][index][den_id] = sgd_score

            for i, (data_als, data_sgd) in enumerate(zip(eval_data_als, eval_data_sgd)):
                als_res_df = pd.DataFrame(data_als, columns=self.density_list, index=self.test_job_id_list)
                sgd_res_df = pd.DataFrame(data_sgd, columns=self.density_list, index=self.test_job_id_list)
                als_res_df_list[i].append(als_res_df)
                sgd_res_df_list[i].append(sgd_res_df)

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
    #     res_subscript = self.cost_restriction
    #
    #     for i in range(self.n_epochs):
    #         als_res_df = pd.read_csv(f"{res_dir}/als_res_df_epoch_{i}_{res_subscript}.csv")
    #         sgd_res_df = pd.read_csv(f"{res_dir}/sgd_res_df_epoch_{i}_{res_subscript}.csv")
    #         for ind, density in enumerate(self.density_list):
    #             als_prob = [sum(i <= thres for i in als_res_df[str(density)]) / len(als_res_df) for thres in self.thres_list]
    #             sgd_prob = [sum(i <= thres for i in sgd_res_df[str(density)]) / len(sgd_res_df) for thres in self.thres_list]
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

    # def plot_final(self, res_dir, density):
    #
    #     line_styles = ["-", ":", "-.", ]  # "--",
    #     colors = ["green", "blue", "red", "cyan", "yellow", "pink"]
    #     plt.figure(figsize=(7, 5))
    #     plt.rcParams.update({'font.size': 8})
    #     metrics = ["0.9", "0.5", "0.1", "0.05", "0.01"]
    #     for alg in ["als", "sgd"]:
    #         for index, metric in enumerate(metrics):
    #             # for epoch in range(self.n_epochs):
    #             for size_idx, size in enumerate(["small", "medium", "large"]):
    #                 epoch = 1
    #                 als_res_df = pd.read_csv(f"{res_dir}/{alg}_res_df_epoch_{epoch}_{size}_{metric}.csv")
    #                 als_prob = [sum(i <= thres for i in als_res_df[str(density)]) / len(als_res_df) for thres in self.thres_list]
    #                 als_prob[0] = min(0.1, als_prob[0])
    #                 # plt.plot(self.thres_list, als_prob, line_styles[epoch % 4], label=f'{metric}' if epoch == 1 else '_nolegend_', color=colors[index])
    #                 plt.plot(self.thres_list, als_prob, line_styles[size_idx], label=f'{metric}_{size}', color=colors[index])
    #                 plt.grid(True)
    #                 # plt.xlabel(f"{alg}")
    #                 plt.xticks(self.thres_list)
    #                 plt.yticks(self.thres_list)
    #
    #         plt.legend(bbox_to_anchor=(.78, .01, .22, .1), loc='lower left',
    #                    ncol=1, mode="expand", borderaxespad=0.1)
    #         plt.tight_layout(h_pad=1)
    #         plt.savefig(f"{res_dir}/fig_complete_{alg}.png")
    #         plt.clf()


if __name__ == '__main__':
    start = time.time()
    for cost_rest in [0.9, 0.5, 0.1, 0.05, 0.01]:
        benchmark = Benchmark(train_size=99, cost_restriction=cost_rest)
        # benchmark.debug("eval")
        benchmark.run_evaluation(f"eval/cost_restriction", "eval/cost_restriction", 10)
        # benchmark.plot_result(f"eval/cost_restriction")
        # print(benchmark.compare_algorithms())

    print(f"Elapsed time: {(time.time() - start) / 60} min")

