from pathlib import Path
import sys

def _make_set(lst)->set:
    s = set()
    for i in lst:
        uni1,uni2 = i.split('_')
        s.add(uni1)
        s.add(uni2)
    return(s)


def read_cluster_outfile(inputFile)->dict:
    out_dict = {}
    with open(inputFile) as oF:
        for line in oF:
            line = line.strip('\n')
            if not line.startswith('# total clusters'):
                lst_stuff= line.split(' ')
                if int(lst_stuff[1]) > 1:
                    value_set = _make_set(lst_stuff[2:])
                    out_dict[lst_stuff[0]]=value_set
    return out_dict

# setting working dir
working_dir = Path('/Users/dc1321/testing_clusters_biological_relevance')
input_filename = sys.argv[1] # amc_cluster_file
inputFile = working_dir/input_filename
print(inputFile)
out_dict = read_cluster_outfile(inputFile)
print(out_dict)
input_filename1 = sys.argv[2] # david_clusters
inputFile1 = working_dir/input_filename
print(inputFile)





