import matplotlib.pyplot as plt
import numpy as np
def read_data(inFile):

    data_dict = {}
    with open(inFile) as oF:
        for line in oF:
            if not line.startswith('Silhouette index for each cluster:'):
                line=line.strip('\n')
                index, list_str = line.split(':')
                index1, si_vals = list_str.split(',')
                lst_si = si_vals.split()
                # perform conversion from str to float
                test_lst = [float(i) for i in lst_si]
                if int(index1) > 10:
                    if index not in data_dict:
                        data_dict[index] = test_lst
   # print(data_dict.values())

    labels, data = data_dict.keys(), data_dict.values()
    #plt.boxplot(data) # simple box plot

    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    bp = plt.boxplot(data, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    plt.xticks(range(1, len(labels) + 1), labels)
    for i in labels:
        size_cluster = len(data_dict[i])
        print(size_cluster)
    plt.xlabel('Cluster #')
    plt.ylabel('Silhouette Indices')
    plt.title('Clusters with size > 10 elements')
    plt.show()

inputfile = "si_scores_sample.txt"
read_data(inputfile)
