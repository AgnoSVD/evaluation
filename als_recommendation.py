import numpy as np
from copy import deepcopy
import logging


class ALS:
    def __init__(self, X, lamb_u, lamb_v, K, max_epoch=100, none_val=99):
        self.X = X
        self.K = K
        self.lamb_u = lamb_u
        self.lamb_v = lamb_v
        self.max_epoch = max_epoch
        self.N = np.shape(self.X)[0]
        self.M = np.shape(self.X)[1]
        self.U = np.random.randn(self.N, self.K)
        self.V_T = np.random.randn(self.K, self.M)
        self.none_val = none_val
        self.log = logging.getLogger('ALS')

    def update_V(self):
        for i in range(len(self.V_T[0])):  # Updating column-wise as V_T is transpose
            rated = np.where(self.X[:, i] != self.none_val)
            sum_u = self.U[rated].T.dot(self.U[rated])
            sum_x = np.zeros(self.K)
            for j in np.ravel(rated):
                sum_x += self.U[j] * self.X[j][i]
            self.V_T[:, i] = np.dot(np.linalg.inv(sum_u + self.lamb_u * np.identity(self.K)), sum_x)

    def update_U(self):
        for i in range(len(self.U)):
            rated = np.where(self.X[i, :] != self.none_val)
            sum_v = self.V_T.T[rated].T.dot(self.V_T.T[rated])
            sum_x = np.zeros(self.K)
            for j in np.ravel(rated):
                sum_x += self.V_T[:, j] * self.X[i][j]
            self.U[i] = np.dot(np.linalg.inv(sum_v + self.lamb_v * np.identity(self.K)), sum_x)

    def calc_risk(self, X):
        est_X = self.U.dot(self.V_T)
        org_X = deepcopy(X)
        unrated_ind = np.where(X == self.none_val)
        org_X[unrated_ind] = est_X[unrated_ind]
        rms_err = np.sqrt(np.sum(np.square(org_X - est_X)) / (self.M * self.N - len(unrated_ind)))
        return rms_err

    def train(self):
        prev_risk = self.calc_risk(self.X)
        conv = 0
        epoch = 0
        while True:
            epoch += 1
            self.update_V()
            self.update_U()
            risk = self.calc_risk(self.X)
            if abs(risk - prev_risk) < 0.000000001:
                conv += 1
            else:
                conv = 0
            if conv == 10:
                break
            if epoch == self.max_epoch:
                break
            prev_risk = risk
        self.log.info(f"converged at rmse: {self.calc_risk(self.X)}, epoch: {epoch}")

    def get_est_matrix(self):
        return self.U.dot(self.V_T)


def train_val_test(data, none_val):
    print("Non-empty entry shape: ", np.shape(np.where(data != none_val)))
    train_data = np.ones(shape=data.shape) * none_val
    val_data = np.ones(shape=data.shape) * none_val
    test_data = np.ones(shape=data.shape) * none_val
    for i in range(len(data)):
        rated_ind = np.ravel(np.where(data[i] != none_val))

        tot_rated = len(rated_ind)
        train_size = int(tot_rated * 0.65) + 1
        train_ind = np.random.choice(rated_ind, size=train_size)
        train_data[i, train_ind] = data[i, train_ind]

        rated_ind = np.setdiff1d(rated_ind, train_ind)
        tot_rated = len(rated_ind)
        val_size = int(tot_rated * 0.6) + 1
        val_ind = np.random.choice(rated_ind, size=val_size)
        val_data[i, val_ind] = data[i, val_ind]

        rated_ind = np.setdiff1d(rated_ind, val_ind)
        tot_rated = len(rated_ind)
        test_size = tot_rated
        test_ind = np.random.choice(rated_ind, size=test_size)
        test_data[i, test_ind] = data[i, test_ind]

    print("Train shape: ", np.shape(np.where(train_data != none_val)))
    print("Validation shape: ", np.shape(np.where(val_data != none_val)))
    print("Test shape: ", np.shape(np.where(test_data != none_val)))

    return train_data, val_data, test_data


if __name__ == '__main__':
    import pandas as pd

    lambda_v = [0.01, 0.1, 1.0, 10]
    lambda_u = [0.01, 0.1, 1.0, 10]
    K = [5, 10, 20, 40]

    dataset_path = '/home/himel/Documents/Academic/datasets/jester_dataset_1_1.csv'  # 'dataset/rating_matrix'
    dataset = pd.read_csv(dataset_path, header=None, sep=',')

    dataset = dataset.drop(0, axis=1)

    data = dataset.values

    # print("data shape: ", np.shape(data))
    # print("Total rating: ", np.shape(np.where(data!=99))[1])
    #
    # reduced_size = 1000
    # X = data[:reduced_size, :]
    # # train_data, val_data, test_data = train_val_test(X, 99)
    #
    # reduced_data = np.ones(shape=X.shape) * 99
    # for i in range(len(X))[:30]:
    #     rated_ind = np.ravel(np.where(X[i] != 99))
    #     tot_rated = len(rated_ind)
    #     val_size = int(tot_rated * 0.1) + 1
    #     val_ind = np.random.choice(rated_ind, size=val_size)
    #     reduced_data[i, val_ind] = data[i, val_ind]
    # print("Reduced data: ", np.shape(np.where(reduced_data!=99))[1])
    # print("K=5, lamb_u=10, lamb_v=10")
    # # model = ALS(X=train_data, K=5, lamb_u=10, lamb_v=10, none_val=99)
    # model = ALS(X=reduced_data, K=5, lamb_u=10, lamb_v=10, none_val=99)
    # model.train()
    #
    # # risk =model.calc_risk(val_data)
    # np.savetxt('dataset/temp_est',model.get_est_matrix())
    # # print("risk: ", risk)

    train_size = 1000
    X = data[:train_size, 1:]

    train_data, val_data, test_data = train_val_test(X, 99)

    models = []
    i = 1
    print("training stage")
    for l_v in lambda_v:
        for l_u in lambda_u:
            for k in K:
                print("model: ", i)
                print("lambda_v: ", l_v, "lambda_u: ", l_u, "K: ", k)
                model = ALS(X=train_data, K=k, lamb_u=l_u, lamb_v=l_v, none_val=99)
                model.train()
                models.append(model)
                i += 1

    print("\n\nvalidation stage")
    model_best = models[0]
    curr_risk = 10
    model_id = 0
    i = 1
    for model in models:
        risk = model.calc_risk(val_data)
        print("model: ", i, "risk: ", risk)

        if risk < curr_risk:
            model_best = model
            curr_risk = risk
            model_id = i
        i += 1
    print("model: ", model_id, "is best")

    print("\n\ntesting stage")
    print("RMSE of test data: ", model.calc_risk(test_data))
