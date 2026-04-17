import numpy as np
import scipy as sp

def calculate_mean(D):
    return D.mean(1) # calculates the mean column-wise

def center_data(D, mu):
    # Now we center data (remove mu form all points)
    # Firstly we reshape mu to be a column vector
    mu = mu.reshape((mu.size, 1))
    return D - mu

def compute_SWC(D, L, val):
    summed = np.zeros((D.shape[0], D.shape[0]))
    summedSb = np.zeros((D.shape[0], D.shape[0]))
    N = D.shape[1]
    global_mean = calculate_mean(D)
    global_mean = global_mean.reshape((global_mean.size, 1))
    for i in np.unique(L):
        dcls = D[:, L==i]
        mean = calculate_mean(dcls)
        DCs = center_data(dcls, mean)
        L1 = list(L)
        n = L1.count(i)
        SWc = (DCs @ DCs.T) / float(n)
        # print("Class: ", i)
        # print("/n")
        summed += (SWc * n)

        diffMeans = mean.reshape((mean.size, 1)) - global_mean
        summedSb += (float(n) * (diffMeans @ diffMeans.T))

    
    SW = summed / float(N)
    SB = summedSb / float(N)
    print("SW: ")
    print("/n")
    print(SW)
    print("SB:")
    print(SB)
    return SW, SB

def train_lda(D, L, no_clases, direction = 1):
    SW, SB = compute_SWC(D, L, val=no_clases)
    s, U = sp.linalg.eigh(SB, SW)
    
    W = U[:, ::-1][:, 0:direction]
    W=-W

    DP = np.dot(W.T, D)

    return DP, W    