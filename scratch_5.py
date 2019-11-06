from pathlib import Path
import itertools as itts
from operator import itemgetter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from sklearn import cluster, datasets, mixture

import scipy
from scipy.sparse import csgraph
# from scipy.sparse.linalg import eigsh
from numpy import linalg as LA

main_dir = Path("/Users/dc1321/humanPPI")
data_file = main_dir/"NW_seqid_complex.txt"


def read_data(inFile) -> pd.DataFrame:
    """Conver file to pandas dataframe
    Arguments:
        inFile {file path}
    Returns:
        [pd.DataFrame] -- [similarity matrix]
    """
    def clean_line(x: str): return x.strip().split()
    data_dict = {}
    with open(inFile) as oF:
        for coins in map(clean_line, itts.islice(oF, 0, None)):
            comp1, comp2, value = coins
            if comp1 not in data_dict:
                data_dict[comp1] = {}
            data_dict[comp1][comp2] = {
                'value': float(value), 'x': None, 'y': None}
            if comp2 not in data_dict:
                data_dict[comp2] = {}
            data_dict[comp2][comp1] = {
                'value': float(value), 'x': None, 'y': None}
            data_dict[comp1][comp1] = {
                'value': 1.0, 'x': None, 'y': None}
            data_dict[comp2][comp2] = {
                'value': 1.0, 'x': None, 'y': None}
    keys = sorted(data_dict.keys())
    for ix, k1 in enumerate(keys):
        for iy, k2 in enumerate(keys):
            data_dict[k1][k2].update(dict(x=ix, y=iy))
    Y = itemgetter('y')
    M = pd.DataFrame(
        [[x['value']
            for x in sorted(data_dict[k].values(), key=Y)] for k in keys],
        index=keys, columns=keys)
    return M


data = read_data(data_file)

"""

# Setup cluster parameters
ward = cluster.AgglomerativeClustering(
    n_clusters=2, linkage='ward',
    connectivity=data.values)
data_cl = ward.fit(data)
L = [[] for x in range(2)]
for key, cl in zip(data.columns, data_cl.labels_):
    L[cl].append(key)
print('# printing cluster')
for ix, cl in enumerate(L, 1):
    str_out = f'{ix} {len(cl)}'
    for el in cl:
        str_out += f' {el}'
    print(str_out)
print(f"Total cluster {len(L)}")
"""


def build_distance_matrix(data: pd.DataFrame, cutoff):
    def dist(x):
        if x > cutoff:
            return(1/x) # use simpliest method
        else:
            return (0)
   # data_out = np.matrix(data.values)
    data_out = np.vectorize(dist)(data.values)
    np.fill_diagonal(data_out, 0)
    return data_out

def getAffinityMatrix(data, cutoff=0.3):
    """
    Calculate affinity matrix based on input similarity matrix and cutoff

    Apply local scaling based on the nearest neighbours based on the threshold/cutoff
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate distance matrix
    dists = build_distance_matrix(data,cutoff)

    # for each row, sort the distances ascending order and take the index of the
    # k-th position (nearest neighbour) u need some k-nearest cutoff see FIG 2 and Formula 2 from the paper atop
    # they say this K is dependenf on the sort of data and dimention you have so u need to play with it
    # this K is needed to calculate local Scaling 
    knn_distances = np.sort(dists, axis=0)
    knn_distances = knn_distances[np.newaxis].T 

    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix



def eigenDecomposition(A, plot=True, topK=2):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
    #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eig(L)
    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()

        # Identify the optimal number of clusters as the index corresponding
        # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors

affinity_matrix = getAffinityMatrix(data, cutoff=0.3)
nb_clusters, eigenvalues,eigenvectors = eigenDecomposition(affinity_matrix)
print(f'Optimal number of clusters {nb_clusters}')

plt.show()
