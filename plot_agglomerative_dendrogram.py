"""
=========================================
Plot Hierarchical Clustering Dendrogram
=========================================
This example plots the corresponding dendrogram of a hierarchical clustering
using Agglomerative Clustering and the dendrogram method available in scipy
The one in sklearn doesn't work!

"""

import itertools as itts
from pathlib import Path
from operator import itemgetter
from optparse import OptionParser, OptionValueError


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

import numpy as np
import pandas as pd


# make it fancy!
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

"""
def dist_matrix_to_1d(M):
    A =[]
    for ix,  row in enumerate(M[:-1]):
        for iy, val in enumerate(row[ix+1:], ix+1):
            A.append(val)
    return np.array(A)
"""

# Create linkage matrix and then plot the dendrogram

def plot_dendrogram(model, threshold):

    plt.title('Hierarchical Clustering Dendrogram')

    # Plot the corresponding dendrogram
    # we can use cut-off to cluster/colour the histogram
    # Break into clusters based on cutoff

    ind = sch.fcluster(model, threshold, 'distance')
    #dendrogram(model, orientation='right', color_threshold=threshold) # show the whole tree
    max_display_levels=10
    fancy_dendrogram(model, truncate_mode='lastp', p=max_display_levels, max_d = threshold)
    plt.show()

def _check_inputFile(option, opt_str, value, parser):
    f_path = Path(value)
    if not f_path.is_file():
        raise OptionValueError(f"Cannot get {str(f_path)} file")
    setattr(parser.values, option.dest, Path(f_path))
    parser.values.saved_infile = True

def read_data(inFile) -> pd.DataFrame:
    """
    Convert file to pandas dataframe
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

    options_parser.add_option("-c", "--cutoff",
                              dest="cutoff", type='float',
                              help="clustering cutoff",
                              metavar="FLOAT")
    (options, args) = options_parser.parse_args()
    in_file = Path(options.input_file)
    cutoff = float(options.cutoff)

    data = read_data(in_file)
    dist = build_distance_matrix(data)
    threshold = 1/(cutoff*cutoff)

    data1D = ssd.squareform(dist)

    dist_test = linkage(data1D, method='complete')     # Complete linkage
                                                        # maximum linkage uses
                                                        # the maximum distances between all observations of the two sets

    plot_dendrogram(dist_test,threshold)



