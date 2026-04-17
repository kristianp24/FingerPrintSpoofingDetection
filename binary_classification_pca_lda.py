from lda import train_lda
from implementation_pca_lda import train_pca, load, plot_simple_hist
from binary_classification_lda import split_db_2to1
import numpy as np

D, L = load("FingerPrintSpoofingDetection/data/trainData.txt")
(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

DP_pca, P = train_pca(DTR, LTR, 2)

DP_lda, W = train_lda(DP_pca, LTR, no_clases=2)

print(DP_lda)

#plot_simple_hist(DP_lda, LTR, "LDA on PCA preprocessing TRAINING DATA")

DVAL_pca = np.dot(P.T, DVAL)
DVAL_lda = np.dot(W.T, DVAL_pca)

plot_simple_hist(DVAL_lda, LVAL, "LDA on PCA preprocessing VALIDATION DATA")

PRED_LABELS = np.zeros(LVAL.shape, np.int32)
m0 = DP_lda[0, LTR == 0].mean()
m1 = DP_lda[0, LTR == 1].mean()
threshold = (m1 + m0) / 2.0

if m1 > m0:
    # Standard orientation: Class 1 is the 'greater' side
    PRED_LABELS[DVAL_lda[0] >= threshold] = 1
    PRED_LABELS[DVAL_lda[0] < threshold] = 0
else:
    # Flipped orientation: Class 1 is the 'lower' side
    PRED_LABELS[DVAL_lda[0] < threshold] = 1
    PRED_LABELS[DVAL_lda[0] >= threshold] = 0

errors = 0
for i in range(len(LVAL)):
        if LVAL[i] != PRED_LABELS[i]:
            errors += 1
    
print("Total labels: ", len(LVAL))
print("Number of errors: ", errors)


