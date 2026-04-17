from lda import train_lda
import numpy 
from implementation_pca_lda import load, plot_simple_hist

def split_db_2to1(D, L, seed=1):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)


if __name__ == "__main__":
    D, L = load("FingerPrintSpoofingDetection/data/trainData.txt")

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    DTR_proj, W = train_lda(DTR, LTR, 2, 1)

    DVAL_proj = numpy.dot(W.T, DVAL)

    plot_simple_hist(DVAL_proj, LVAL, "LDA for validation set")
    plot_simple_hist(DTR_proj, LTR, "LDA for training set")

    mean_0 = DTR_proj[0, LTR==0].mean()
    mean_1 = DTR_proj[0, LTR==1].mean()
    threshold = (mean_0 + mean_1) / 2.0

    print("Threshold: ", threshold)

    pred_labels = numpy.zeros(LVAL.shape, numpy.int32)
    pred_labels[DVAL_proj[0] >= threshold] = 1
    pred_labels[DVAL_proj[0] < threshold] = 0

    print("Original Lables from validation: \n")
    print(LVAL)
    print("Predicted Labels: \n")
    print(pred_labels)

    errors = 0
    for i in range(len(LVAL)):
        if LVAL[i] != pred_labels[i]:
            errors += 1
    
    print("Total labels: ", len(LVAL))
    print("Number of errors: ", errors)
