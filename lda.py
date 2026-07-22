import numpy as np
import scipy as sp

class LDA:
    def __init__(self, Data, L, no_clases, direction=1):
        self.D = Data
        self.L = L
        self.no_clases = no_clases
        self.direction = direction

    def calculate_mean(self, D):
        return D.mean(axis=1)

    def center_data(self, D, mu):
        mu = mu.reshape((-1, 1))
        return D - mu

    def compute_SWC(self, D, L):
        num_features, N = D.shape
        summed_SW = np.zeros((num_features, num_features))
        summed_SB = np.zeros((num_features, num_features))
        
        global_mean = self.calculate_mean(D).reshape((-1, 1))

        for i in np.unique(L):
            dcls = D[:, L == i]
            n = dcls.shape[1]
            
            mean_cls = self.calculate_mean(dcls).reshape((-1, 1))
            DCs = self.center_data(dcls, mean_cls)
            
            summed_SW += (DCs @ DCs.T)

            diff_means = mean_cls - global_mean
            summed_SB += float(n) * (diff_means @ diff_means.T)

        SW = summed_SW / float(N)
        SB = summed_SB / float(N)
        
        return SW, SB

    def train_lda(self):
        SW, SB = self.compute_SWC(self.D, self.L)
        
        s, U = sp.linalg.eigh(SB, SW)
        
        W = U[:, ::-1][:, 0:self.direction]

        DP_temp = np.dot(W.T, self.D)
        if DP_temp[0, self.L == 1].mean() < DP_temp[0, self.L == 0].mean():
            W = -W

        DP = np.dot(W.T, self.D)

        return DP, W