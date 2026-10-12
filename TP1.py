import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import combinations


chemin_donnees = './gene+expression+cancer+rna+seq/data.csv'
chemin_labels = './gene+expression+cancer+rna+seq/labels.csv'

def charger_donnees() :
    donnees = pd.read_csv(chemin_donnees, sep=',', index_col=0)
    labels = pd.read_csv(chemin_labels, sep=',', index_col=0).iloc[:, 0]
    return donnees, labels


def dist_intra(C, euclid=True) :
    x_c = np.mean(C, axis=0)
    if euclid == True :
        dist = cdist(C, [x_c], metric='euclidean')
    else :
        mat_cov = np.cov(C, rowvar=False)
        inv_cov = np.linalg.pinv(mat_cov)
        dist = cdist(C, [x_c], metric='mahalanobis', VI=inv_cov)
    return np.max(dist)

def dist_inter(C1, C2, euclid=True) :
    x_c1 = np.mean(C1, axis=0)
    x_c2 = np.mean(C2, axis=0)
    if euclid == True :
        d1 = np.min(cdist(C1, [x_c2], metric='euclidean'))
        d2 = np.min(cdist(C2, [x_c1], metric='euclidean'))
    else :
        C12 = np.vstack([C1, C2])
        mat_cov = np.cov(C12, rowvar=False)
        inv_cov = np.linalg.pinv(mat_cov)
        d1 = np.min(cdist(C1, [x_c2], metric='mahalanobis', VI=inv_cov))
        d2 = np.min(cdist(C2, [x_c1], metric='mahalanobis', VI=inv_cov))
    return min(d1, d2)

def overlap(C1, C2, euclid=True) :
    return (dist_intra(C1, euclid) + dist_intra(C2, euclid))/(2*dist_inter(C1, C2, euclid))

def tab_overlap(donnees, labels, euclid=True) :
    classes = np.unique(labels)
    mat = np.zeros((len(classes), len(classes)))
    tab = pd.DataFrame(mat, index=classes, columns=classes, dtype=float)
    for c1, c2 in combinations(classes, 2):
        C1 = donnees[labels == c1].values
        C2 = donnees[labels == c2].values
        val = overlap(C1, C2, euclid)
        tab.loc[c1, c2] = val
        tab.loc[c2, c1] = val
    return tab


def main() :
    donnees, labels = charger_donnees()
    tab = tab_overlap(donnees, labels)
    print(tab)
    return

main()