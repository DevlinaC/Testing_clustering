# Python program to print connected
# components in an undirected graph

import itertools as itts
from pathlib import Path
from optparse import OptionParser, OptionValueError


import networkx as nx
import matplotlib.pyplot as plt


def _check_inputFile(option, opt_str, value, parser):
    f_path = Path(value)
    if not f_path.is_file():
        raise OptionValueError(f"Cannot get {str(f_path)} file")
    setattr(parser.values, option.dest, Path(f_path))
    parser.values.saved_infile = True

def read_data(inFile, num_lines: int = None):
    out_dict = {}
    with open(inFile) as oF:
        for ix, line in enumerate(itts.islice(oF, 0, num_lines)):
            line = line.strip()
            try:
                domain1, domain2, tm = line.split()
            except ValueError:
                print(f'wrong line {ix}')
                continue
            value = float(tm)
            if domain1 == domain2:
                continue
            if domain1 not in out_dict:
                out_dict[domain1] = {}
            out_dict[domain1][domain2] = dict(value=value)
    return out_dict


def make_graph(data_dict, cutoff=0.6):
    G = nx.Graph()
    nodes = [(n, dict(uid=n, label=n)) for n in data_dict.keys()]
    G.add_nodes_from(nodes)
    for node, v in data_dict.items():
        edges = [(node, x) for x in filter(
            lambda x: x in data_dict.keys(),  v.keys()) if v[x]['value'] >= cutoff]
        G.add_edges_from(edges)
    return G

def num_nodes(x):
    return(nx.number_of_nodes(x))

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
    options_parser.add_option("-o", "--out_file",
                              dest="out_file", type='str',
                              help="output FILE saves in graphml format",
                              metavar="FILE")

    (options, args) = options_parser.parse_args()

    in_file = Path(options.input_file)
    out_file = Path(options.out_file)
    start_graph_file = out_file.parent / f"{out_file.stem}_start.graphml"
    initial_graph = out_file.parent / f"{out_file.stem}_initial_png"
    cutoff = float(options.cutoff)

    data = read_data(in_file)
    G = make_graph(data, cutoff)
    nx.write_graphml(G, str(start_graph_file))
    nx.draw(G)
    plt.savefig(initial_graph)

    for ix, g in enumerate(
            nx.connected_component_subgraphs(G),  1):
        str_out = ""
        str_out += f"{ix} {num_nodes(g)}"
        for node, data in g.nodes.items():
            str_out += f" {data['label']}"
            print(str_out)
    print(f"# total cluster {ix}")

