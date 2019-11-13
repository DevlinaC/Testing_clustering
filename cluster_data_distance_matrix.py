"""
Agglomerative Clustering
Recursively merges the pair of clusters that minimally increases a given linkage distance

"""

import itertools as itts
from pathlib import Path
from operator import itemgetter
from optparse import OptionParser, OptionValueError

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score

def dist_matrix_to_1d(M):
    A =[]
    for ix,  row in enumerate(M[:-1]):
        for iy, val in enumerate(row[ix+1:], ix+1):
            A.append(val)
    return np.array(A)


def _check_inputFile(option, opt_str, value, parser):
    f_path = Path(value)
    if not f_path.is_file():
        raise OptionValueError(f"Cannot get {str(f_path)} file")
    setattr(parser.values, option.dest, Path(f_path))
    parser.values.saved_infile = True

def read_data(inFile) -> pd.DataFrame:
    """Convert file to pandas dataframe
    Arguments:
        inFile {file path}
    Returns:
        [pd.DataFrame] -- [similarity matrix]
    """
    def clean_line(x: str): return x.strip().split()
    data_dict = {}
    with open(inFile) as oF:
        for coins in map(clean_line, itts.islice(oF, 0, None)):
            pdb1, pdb2, value = coins
            if pdb1 not in data_dict:
                data_dict[pdb1] = {}
            data_dict[pdb1][pdb2] = {
                'value': float(value), 'x': None, 'y': None}
            if pdb2 not in data_dict:
                data_dict[pdb2] = {}
            data_dict[pdb2][pdb1] = {
                'value': float(value), 'x': None, 'y': None}
            data_dict[pdb1][pdb1] = {
                'value': 1.0, 'x': None, 'y': None}
            data_dict[pdb2][pdb2] = {
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

def build_distance_matrix(data: pd.DataFrame):
    def dist(x): return 1.0/(x*x)
    data_out = np.vectorize(dist)(data.values)
    np.fill_diagonal(data_out, 0)
    return data_out


if __name__ == "__main__":
    options_parser = OptionParser()
    options_parser.add_option("-i", "--input_file",
                              dest="input_file", type='str',
                              help="input FILE",
                              metavar="FILE",
                              action='callback',
                              callback=_check_inputFile)
    options_parser.add_option("-o", "--out_file",
                              dest="out_file", type='str',
                              help="output FILE",
                              metavar="FILE")
    options_parser.add_option("-c", "--cutoff",
                              dest="cutoff", type='float',
                              help="clustering cutoff",
                              metavar="FLOAT")
    (options, args) = options_parser.parse_args()
    in_file = Path(options.input_file)
    out_file = Path(options.out_file)
    cutoff = float(options.cutoff)


    data = read_data(in_file)
    dist = build_distance_matrix(data)
    threshold = 1/(cutoff*cutoff)
    dist_test = cluster.AgglomerativeClustering(
        n_clusters=None,
        affinity='precomputed',
        distance_threshold=threshold,
        linkage='complete')     # Maximum or complete linkage
                                # minimizes the maximum distance between
                                # observations of pairs of clusters

    data_cl = dist_test.fit(dist)
    #cluster_labels = clusterer.fit_predict(X)
    L = [[] for x in range(data_cl.n_clusters_)]

    # getting labels for printing and to calculate silhouette score #
    labels = data_cl.labels_

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    ## NEED TO USE DISTANCE NOT SIMILARITY
    '''
   The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster DISTANCE (b) for each sample.  
    '''
    silhouette_avg = silhouette_score(dist, labels, metric='precomputed')
    print("For n_clusters =", len(L),
          "The average silhouette_score is :", silhouette_avg)
    for key, cl in zip(data.columns, labels):
        L[cl].append(key)
    oF = open(out_file, 'w')
    nb_clusters = len(L)
    oF.write(f"# total clusters {nb_clusters} , average silhouette_coefficient {silhouette_avg} \n")
    for ix, cl in enumerate(L, 1):
        str_out = f"{ix} {len(cl)}"
        for el in cl:
            str_out += f" {el}"
        oF.write(str_out+'\n')
    oF.close()



    # Compute the silhouette scores for each sample/cluster
    sample_silhouette_values = silhouette_samples(dist, data_cl.labels_, metric="precomputed")
    print("Silhouette index for each cluster:")
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    for i in range(nb_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        print(size_cluster_i,ith_cluster_silhouette_values)

    array_1d = dist_matrix_to_1d(dist)
