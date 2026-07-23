import numpy
import scipy
from gaussian import logpdf_GAU_ND
from lda import LDA


class StandardMVG:
    def __init__(self, D, L, DTE, LTE):
        self.D = D
        self.L = L
        self.DTE = DTE
        self.LTE = LTE
        self.S = None

    def mean(self, X):
        return numpy.mean(X, axis=1, keepdims=True)

    def calculate_C(self, mu, X):
        DC = X - mu
        return (DC @ DC.T) / float(X.shape[1])

    def calculate_mean_cov_per_class(self, DTR, LTR):
        unique_labels = numpy.unique(LTR)
        parameters = []
        for label in unique_labels:
            data_class = DTR[:, LTR == label]
            mu = self.mean(data_class)
            C = self.calculate_C(mu, data_class)
            parameters.append((mu, C))
        
        return parameters

    def vcol(self, x):
        return x.reshape((-1, 1))

    def vrow(self, x):
        return x.reshape((1, -1))

    def compute_logPosterior(self, S_logLikelihood, v_prior):
        SJoint = S_logLikelihood + self.vcol(numpy.log(v_prior))
        SMarginal = self.vrow(scipy.special.logsumexp(SJoint, axis=0))
        SPost = SJoint - SMarginal
        return SPost

    def apply_MVG(self):
        parameters = self.calculate_mean_cov_per_class(self.D, self.L)
        
        scores = [logpdf_GAU_ND(self.DTE, mean, cov) for mean, cov in parameters]
        self.S = numpy.vstack(scores)
        
        num_classes = len(parameters)
        priors = numpy.ones(num_classes) / float(num_classes)
        
        S_logPost = self.compute_logPosterior(self.S, priors)
        PVAL = S_logPost.argmax(0)
        
        error_rate = (PVAL != self.LTE).sum() / float(self.LTE.size) * 100
        print(f"{self.__class__.__name__} - Error rate: {error_rate:.1f}%")    

        return PVAL


class NaiveBayesMVG(StandardMVG):
    def calculate_mean_cov_per_class(self, DTR, LTR):
        unique_labels = numpy.unique(LTR)
        parameters = []
        for label in unique_labels:
            data_class = DTR[:, LTR == label]
            mu = self.mean(data_class)
            C = self.calculate_C(mu, data_class)
            
            C_naive = C * numpy.eye(C.shape[0])
            parameters.append((mu, C_naive))
        
        return parameters
    


class TiedMVG(StandardMVG):
    def calculate_mean_cov_per_class(self, DTR, LTR):
        lda = LDA(DTR, LTR, no_clases=len(numpy.unique(LTR)))
        SW, _ = lda.compute_SWC(DTR, LTR)
        
        unique_labels = numpy.unique(LTR)
        parameters = []
        for label in unique_labels:
            data_class = DTR[:, LTR == label]
            mu = self.mean(data_class)
            parameters.append((mu, SW))
            
        return parameters

