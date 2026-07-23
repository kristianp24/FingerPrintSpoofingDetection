import numpy as np

# X = np.load("XND.npy")


def mean(X):
    mu =  np.mean(X, 1)
    return mu.reshape(X.shape[0], 1)

def calculate_C(mu, X):
    DC = X - mu
    C =  (DC @ DC.T) / float(X.shape[1])
    return C

def calculate_term3_1sample(xi, mu , C):
    centered_data = (xi - mu)
    inverse_c = np.linalg.inv(C)
    term3 = -0.5 * centered_data.T @ inverse_c @ centered_data
    return term3

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    M_slash_2 = -1 * (M / 2)

    term1 = M_slash_2 * np.log(2 * np.pi)
    
    sign, logC = np.linalg.slogdet(C)
    term2 = -0.5 * logC

    y = []
    for i in range(x.shape[1]):
        xi = x[:, i:i+1]  # this way so we keep x1 a column vector
        term3 = calculate_term3_1sample(xi, mu, C)
        n = term1 + term2 + term3
        y.append(n)
    
    return np.array(y).ravel()

def loglikelihood(X, mu_ML, C_ML):
    scores = logpdf_GAU_ND(X, mu_ML, C_ML)
    return np.sum(scores)

# mu_Ml = mean(X)
# C_ML = calculate_C(mu_Ml, X)
# print(mu_Ml)
# print("Daca te uiti esti bg")
# print(C_ML)
# ll = loglikelihood(X, mu_Ml, C_ML)
# print(ll)

# import matplotlib.pyplot as plt
# plt.figure()
# XPlot = np.linspace(-8, 12, 1000).reshape(1, -1)
# m = np.ones((1,1)) * 1.0
# C = np.ones((1,1)) * 2.0
# plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot, m, C)))
# plt.show()
# pdfSol = np.load('llGAU.npy')
# pdfGau = logpdf_GAU_ND(XPlot, m, C)
# print(np.abs(pdfSol - pdfGau).max())
# print("Second solution \n")
# XND = np.load('XND.npy')
# mu = np.load('muND.npy')
# C = np.load('CND.npy')
# pdfSol = np.load('llND.npy')
# pdfGau = logpdf_GAU_ND(XND, mu, C)
# print(np.abs(pdfSol - pdfGau).max())
