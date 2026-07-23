from gaussian_classifier import StandardMVG, NaiveBayesMVG, TiedMVG
from utils import load, split_db_2to1

D, L = load("data/trainData.txt")
(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

standardMVG = StandardMVG(DTR, LTR, DVAL, LVAL)
standardMVG.apply_MVG()

naiveBayesMVG = NaiveBayesMVG(DTR, LTR, DVAL, LVAL)
naiveBayesMVG.apply_MVG()

tiedMVG = TiedMVG(DTR, LTR, DVAL, LVAL)
tiedMVG.apply_MVG()