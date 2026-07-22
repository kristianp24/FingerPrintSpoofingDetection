import numpy as np
import scipy

class PCA:
    def __init__(self, D, L, components = 2):
        self.D = D
        self.L = L
        self.components = components

    def load(D):
        pass

    def calculate_mean(self):
        return self.D.mean(1)

    def calculate_centered_data(self, mu):
        mu = mu.reshape(mu.size, 1)
        return self.D - mu

    def calculate_covariance(self, DC):
        return (DC @ DC.T) / float(self.D.shape[1])

    def apply_eigen(self, C):
        eigenValues, eigenVectors = np.linalg.eigh(C)
        print("Eigen values: s: ", eigenValues)
        print("Eigen vector: ", eigenVectors)
        return eigenValues, eigenVectors  

    def eigen_Vector_Backwards(self, eigenVector):
        return eigenVector[:, ::-1][:, 0:self.components]  

    def make_projection(self, eigenVectors):
        return np.dot(eigenVectors.T, self.D)


    def train_pca(self):
        mu = self.calculate_mean()
        print("Mean:", mu)
        DC = self.calculate_centered_data(mu)
        #print(DC)
        C = self.calculate_covariance(DC)
        #print(C)
        eigenValues, eigenVectors = self.apply_eigen(C)
        # print(eigenValues)
        # print(eigenVectors)
        eigenVectors = self.eigen_Vector_Backwards(eigenVectors)
        #print(eigenVectors)
        DP = self.make_projection(eigenVectors)
        
        
        return DP, eigenVectors
