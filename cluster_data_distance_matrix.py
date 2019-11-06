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


def _check_inputFile(option, opt_str, value, parser):
    f_path = Path(value)
    if not f_path.is_file():
        raise OptionValueError(f"Cannot get {str(f_path)} file")
    setattr(parser.values, option.dest, Path(f_path))
    parser.values.saved_infile = True


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
    data_out = np.matrix(data.values)
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
    start_graph_file = out_file.parent/f"{out_file.stem}_start.graphml"
    end_graph_file = out_file.parent/f"{out_file.stem}_end.graphml"

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
    L = [[] for x in range(data_cl.n_clusters_)]
    for key, cl in zip(data.columns, data_cl.labels_):
        L[cl].append(key)
    oF = open(out_file, 'w')
    for ix, cl in enumerate(L, 1):
        str_out = f"{ix} {len(cl)}"
        for el in cl:
            str_out += f" {el}"
        oF.write(str_out+'\n')
    oF.write(f"# total cluster {ix}\n")
    oF.close()