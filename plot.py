import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from bert.eval import results

import numpy as np
import pandas as pd


def plot_label_dist(cluster,palette='inferno',figsize=(10,5),flip_axes=False):
    
    # Graph for dist of merchants across labels
    
    labels = sorted(cluster.labels_.astype(float))
    Y = [labels.count(x) for x in set(labels)]
    X = list(set(labels))
    
    plt.figure(figsize=figsize)
    if flip_axes: sns.barplot(Y,X,palette=palette)
    else: sns.barplot(X,Y,palette=palette)
    plt.xlabel('Cluster label')
    plt.ylabel('# of member restaurants')

    plt.title('distribution across {} clusters'.format(len(set(labels))))
    plt.show()
    

def plot_clusters(cluster
                  ,pc=None
                  ,text=False
                  ,n_names=10
                  ,centroids=False
                  ,figsize=(6,6)
                  ,multiple_plots=False
                  ,labels=[]
                  ,names=None
                  ,title=''):
    
    # Graph for coloured scatterplot
    
    sns.set()
    pca0 = pc[:,0]
    pca1 = pc[:,1]
    pca2 = pc[:,2]
    
    if len(labels)==0:
        labels = cluster.labels_
    
    if centroids: centroid_plot = cluster.cluster_centers_
    n = 2 if multiple_plots else 1
    
    for i,other_pc in enumerate([pca1,pca2][:n]):
        count=0
        
        plt.figure(figsize=figsize)
        plt.scatter(x=pca0,y=other_pc
                    ,c=labels.astype(float)
                    , s=50
                    , alpha=0.5)
        if centroids: plt.scatter(centroid_plot[:,0], centroid_plot[:,1], c='blue', s=50)
        
        # Add merchant name annontations
        if text:
            for n, (j, x, y) in enumerate(zip(names, pca0, other_pc)):
                if count>n_names: break
                if np.random.rand(1)[0]>0.8:
                    count+=1
                    
                    xytexts = [+3,-3,+5,-5,+7,-7]
                    xco = np.random.choice(xytexts); yco = np.random.choice(xytexts)

                    plt.annotate(j, xy=(x, y),xytext=(x+xco, y+yco),fontsize=10,
                        arrowprops=dict(facecolor=np.random.rand(3), shrink=0.05),)

        plt.title('{}'.format(title))
        plt.show()


def plot_3d_clusters(clusters
                     ,text=False
                     ,n_names=8
                     ,names=None
                     ,pc=None
                     ,figsize=(6,6)):
    
    from mpl_toolkits.mplot3d import Axes3D
    np.random.seed(5)

    X = pc[:,0]
    Y = pc[:,1]
    Z = pc[:,2]

    fignum = 1

    for name,est in clusters:
        count=0
        fig = plt.figure(fignum, figsize=figsize)
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        ax.scatter(X, Y, Z,
                   c=est.labels_.astype(float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('component0')
        ax.set_ylabel('component1')
        ax.set_zlabel('component2')
        ax.set_title(name)
        ax.dist = 12
        fignum = fignum + 1
        
    if text:
        for n, (j, x, y, z) in enumerate(zip(names, X,Y,Z)):
            if count>n_names: break
            if np.random.rand(1)[0]>0.5:
                count+=1

                xytexts = [+3,-3,+5,-5,+7,-7]
                xco = np.random.choice(xytexts); yco = np.random.choice(xytexts)

                plt.annotate(j, xy=(x, y),xytext=(x+xco, y+yco),fontsize=10,
                    arrowprops=dict(facecolor=np.random.rand(3), shrink=0.05),)

    fig.show()   


def plot_pc(embeds,names):
    pca = PCA(n_components=30)
    principal_components = pca.fit_transform(embeds)

    pca0 = principal_components[:,0]
    pca1 = principal_components[:,1]
    pca2 = principal_components[:,2]

    summed = sum(pca.explained_variance_)

    plt.bar(range(len(pca.explained_variance_)),[i/summed for i in (pca.explained_variance_)])
    plt.title('Variance captured per principal component.')
    plt.ylabel('% of variance')
    plt.xlabel('Component number')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(pca0,pca1,c='white')
    for i, x, y in zip(names, pca0[:50], pca1[:50]):
        plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=8)
    plt.xlabel('principal component 0')
    plt.ylabel('principal component 1')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(pca0,pca2,c='white')
    for i, x, y in zip(names, pca0[:50], pca2[:50]):
        plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=8)
    plt.xlabel('principal component 0')
    plt.ylabel('principal component 2')
    plt.show()
    

def minimal_dendrogram(*args, **kwargs):

    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    city = kwargs["city"]
    del kwargs["city"]
    
    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram for {}'.format(city))
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


def plot_metrics(names,cluster_names=['kmeans++'
                 ,'agglomerative_l','agglomerative_m'
                 ,'agglomerative_s','agglomerative_t']):

    data = results(top_k=10,stop_at=100,names=list(names))

    X = range(1,10)

    plt.figure(figsize=(16,8))

    for Y in cluster_names:
        if Y=='affinity_propagation': pass
        else:
            sns.lineplot(y=Y,x=X,data=data[0][1],label=Y,legend='full'
                            ).set(xlabel='top k rankings'
                                ,ylabel='Binary ndcg @ k'
                                ,title='Binary ndcg scores'
                                )

    plt.legend()
    plt.show()

    print(data[0][1])
    return data[0][1]
