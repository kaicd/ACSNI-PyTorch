"""
Name: ACSNI-model
Author: Chinedu A. Anene, Phd
"""
from sklearn.decomposition import PCA, NMF
import warnings
from sklearn import metrics
import numpy as np
from pandas import DataFrame as Dff
import pandas as pd
import torch as th
from torch.utils.data import DataLoader as dl
from tqdm import tqdm
from ACSNI.dat import name_generator
from ACSNI.model import auto_encoder
from ACSNI.dataset import exp_dataset


class DimRed:

    """
    Class for quadruple dimension reduction.
    """

    def __init__(self, x, w, p):
        """
        Dimension reduction class.

        Parameters:
            x: Input matrix (np array)
            w: Weights to adjustment for ae
            p: latent dimension for ae
        """
        self.x = x
        self.w = w
        self.p = p
        self.pca = None
        self.nmf = None
        self.ae = None
        self.__reduced = None
        self.__pcanmf = None
        self.__median = None
        self.__ael1 = None
        self.__ael2 = None
        self.__a = None
        self.__r = 15
        self.__a1 = 0.03
        self.__a2 = 0.85
        self.__scorer = metrics.explained_variance_score
        self.__run_id = name_generator(6)
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    def __str__(self):
        return "Quadruple dimension reduction class"

    def __repr__(self):
        return "\n" + self.__str__()

    def __get_score(self, model, y):
        """
        Determine level of explained variance

        """
        prediction = model.inverse_transform(model.transform(y))
        return self.__scorer(y, prediction)

    def l_med(self):
        self.__median = pd.DataFrame(np.median(self.x.T, axis=1))
        self.__median = self.__median.add_prefix("MEDIAN_" + self.__run_id + "_")
        return

    def lde(self):
        """
        Decompose with PCA and NMF

        """
        self.pca = PCA(n_components=0.95)
        self.pca.fit(self.x)
        pc_weights = pd.DataFrame(self.pca.components_.T)
        pc_weights = pc_weights.add_prefix("PCA_" + self.__run_id + "_")

        opti_rank = []
        #
        warnings.filterwarnings("ignore")
        for k in range(2, self.__r):
            nmf = NMF(n_components=k, max_iter=1000).fit(self.x)
            score_it = self.__get_score(nmf, self.x)
            opti_rank.append(score_it)
            if score_it >= 0.95:
                break

        self.nmf = NMF(n_components=len(opti_rank) + 1, max_iter=10000)
        self.nmf.fit(self.x)
        warnings.resetwarnings()
        #

        nmf_weights = pd.DataFrame(self.nmf.components_.T)
        nmf_weights = nmf_weights.add_prefix("NMF_" + self.__run_id + "_")

        self.__pcanmf = pd.concat([pc_weights, nmf_weights], axis=1)

        return

    def __de4ae(self, y):
        """
        Estimate optimal dimension for AE,
          based on Bahadur and Paffenroth 2020, IEEE

        """
        s_x = y.copy()

        for t in range(s_x.shape[0]):
            s_x.iloc[t, :] = np.sort(np.array(s_x.iloc[t, :]))[::-1]

        svp = np.sort(s_x.mean())[::-1]
        svp_sum = svp.sum()
        alg1 = sum(svp / svp_sum > self.__a1)
        alg2 = 0

        temp = (svp_sum * self.__a2) / 1
        temp2 = 0

        for i in range(len(svp)):
            temp2 += svp[i]
            alg2 += 1

            if temp2 >= temp:
                break
        return int((alg1 + alg2) / 2)

    def __aer(self, nc, epoch):
        """
        Build model structure and run model
        """
        train_loader = dl(
            exp_dataset(self.x),
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.ae = auto_encoder(nc, self.__a).to(self.device)
        self.loss = th.nn.MSELoss()
        self.optimizer = th.optim.RMSprop(
            self.ae.parameters(), lr=1e-3, weight_decay=1e-9
        )

        for _ in tqdm(range(epoch)):
            for _, x in enumerate(train_loader):
                x = x.to(th.float32).to(self.device)
                self.optimizer.zero_grad()
                o = self.ae.forward(x)
                l = self.loss(o, x)
                l.backward()
                self.optimizer.step()
        return

    def mud_ae(self):
        """
        Model output.
        """
        nam = self.__run_id + "_model.pt"
        if self.p == 0:
            self.__a = self.x.shape[0] * 50 // 100
            self.__aer(nc=self.x.shape[1], epoch=500)
            th.save(self.ae.state_dict(), "est_" + nam)
            code_est = Dff(self.ae.encoder[0].weight)
            code_est.to_csv("est.csv")
            self.__a = self.__de4ae(code_est)
            print("The optimal number of dimension is {}".format(self.__a))
        else:
            self.__a = self.p

        self.__aer(nc=self.x.shape[1], epoch=3000)
        th.save(self.ae.state_dict(), nam)
        self.__ael1 = Dff(self.ae.encoder[0].weight.T)
        self.__ael1 = self.__ael1.add_prefix("AE_" + self.__run_id + "_")
        self.__ael2 = Dff(self.ae.encoder[2].weight.T)
        self.__ael2["run"] = nam
        self.__ael2.to_csv("code_{}.csv".format(self.__run_id), index=False)
        return

    def fit(self):
        """
        Fit quadruple dimension reduction {Median, PCA, NMF, AE[DE]}
        """
        self.l_med()
        self.lde()
        self.mud_ae()
        self.__reduced = pd.concat([self.__ael1, self.__pcanmf, self.__median], axis=1)
        return

    def get_reduced(self):
        """
        Get reduced dimension
        """
        return self.__reduced

    def get_aede(self):
        return self.__a

    def add_reduced_row(self, y):
        self.__reduced["ID"] = y
        self.__pcanmf["ID"] = y
        self.__ael1["ID"] = y
        return
