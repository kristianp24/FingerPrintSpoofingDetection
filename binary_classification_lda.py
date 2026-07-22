from lda import LDA
import numpy 
from utils import load, plot_simple_hist, split_db_2to1


if __name__ == "__main__":
    D, L = load("data/trainData.txt")

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    lda = LDA(DTR, LTR, 2, 1)
    DTR_proj, W = lda.train_lda()

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
