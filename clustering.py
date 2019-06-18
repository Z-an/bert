from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent, cophenet
from scipy.spatial.distance import pdist

from sklearn.cluster import KMeans, SpectralClustering

from plot import minimal_dendrogram, plot_clusters, plot_label_dist, plot_3d_clusters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def agglomerative(embeds,names):

    l = linkage(embeds, method='complete', metric='seuclidean')

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('word')
    plt.xlabel('distance')

    dendrogram(
        l,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=0.,  # font size for the x axis labels
        orientation='top',
    )
    plt.show()

    minimal_dendrogram(
        l,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,
        city='Sydney'
    )
    plt.show()

    corr, coph_dists = cophenet(l, pdist(embeds))
    print('\nCophenetic correlation:', corr,'\n')

    return l


def kmeans(principal_components,names,embeds,viz=True):
    kplus = KMeans(n_clusters=12,init='k-means++').fit(embeds)

    if viz:
        plot_clusters(kplus
                    ,pc=principal_components
                    ,text=True
                    ,names=names
                    ,n_names=15
                    ,figsize=(16,4))

        plot_label_dist(kplus,palette='Reds',figsize=(4,2))
        plot_3d_clusters([('Kmeans++',kplus)],pc=principal_components,figsize=(6,4.5))
    
    return kplus


def getFlatLabels(principal_components,
                    model,
                    names,
                    urls,
                    large_cutoff=100,
                    medium_cutoff=75,
                    small_cutoff=50,
                    tiny_cutoff=30,
                    viz=True):

    t_values = [('Large clusters',large_cutoff),('Medium clusters',medium_cutoff),('Small clusters',small_cutoff),('Tiny clusters',tiny_cutoff)]

    agglom_labels = []

    for label, t_value in t_values:
        print(label,'n:')
        clusters = fcluster(model, t=t_value, criterion='distance')
        print(len(np.unique(clusters)),'\n')

        agglom_labels += [clusters]
        
    agglom_labels = np.array(agglom_labels)


    for i in range(len(agglom_labels)): 
        plot_clusters(model,pc=principal_components
                    ,labels=agglom_labels[i]
                    ,names=names
                    ,text=True,figsize=(20,4)
                    ,title=('{} derived from stopping at {} covariance'
                            ).format(t_values[i][0],t_values[i][1]))
        plt.show()
    
    return agglom_labels


def clusters_df_et_embeds_df(names,
                embeds,
                urls,
                cuisines,
                principal_components,
                agglom_labels,
                kplusModel):

    metadata_df = pd.DataFrame(zip(names,urls,cuisines,principal_components[:,0],
                                principal_components[:,1],
                                principal_components[:,2])
                            ,columns=['name','url','cuisines','pca_0','pca_1','pca_2'])

    metadata_df['agglomerative_l'] = agglom_labels[0]
    metadata_df['agglomerative_m'] = agglom_labels[1]
    metadata_df['agglomerative_s'] = agglom_labels[2]
    metadata_df['agglomerative_t'] = agglom_labels[3]
    metadata_df['kmeans++'] = kplusModel.labels_

    embeddings_df = pd.DataFrame(embeds)
    embeddings_df['name'] = names

    return metadata_df, embeddings_df
    
