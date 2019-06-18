from scipy.spatial.distance import pdist, cosine, euclidean

import pandas as pd
import numpy as np

import math
from collections import defaultdict

from clustering import clusters_df_et_embeds_df
from plot import plot_metrics


def nearest_neighbours(embeddings,from_index=None,from_name=None,names=None,to_df=True,cos=True):
    
    if from_index==None and from_name!=None: from_index = names.index(from_name)
        
    from_emb = embeddings[from_index]
    from_name = names[from_index] if from_name==None else from_name
    
    if cosine: dist = lambda x: cosine(from_emb,x)
    else: dist = lambda x: euclidean(from_emb,x)
    
    nn = sorted([(to_name,dist(to_emb)) 
                     for to_emb,to_name 
                         in zip(embeddings,names)]
                
                ,key=lambda x:x[1]
                ,reverse=False)
    
    if to_df: nn = pd.DataFrame(nn,columns=['name','distance']).drop_duplicates()
    
    return from_name,nn



def meta_nearest_neighbours(metadata_df
                            ,embeddings
                            ,names=None
                            ,from_index=None
                            ,from_name=None):
    
    from_item, nn_df = nearest_neighbours(embeddings
                                         ,from_index=from_index
                                         ,from_name=from_name
                                         ,names=names
                                         ,to_df=True)
    
    return from_item, nn_df.merge(metadata_df
                        ).sort_values(by='distance'
                                 ).drop_duplicates(subset='name')
    
def ndcg(ranking
         ,cluster_cardinality
         ,binary=True
         ,receptive_field=1):
    
    dcg = lambda x: sum([rel/math.log(pos+2) for pos,rel in enumerate(x)])
    
    n = len(ranking) if (len(ranking) < cluster_cardinality) else cluster_cardinality
    
    actual_dcg = dcg(ranking)
    ideal_dcg = dcg([1] * n)
    normalized_dcg = actual_dcg / ideal_dcg
    
    return normalized_dcg


def rank_scores(item
                ,nn
                ,binarize=True
                ,k=10
                ,cluster_names=['kmeans++'
                 ,'agglomerative_l','agglomerative_m'
                 ,'agglomerative_s','agglomerative_t']):
    """
    Input: 
        nearest neighbours dataframe,
        k to represent size of list.
    
    Returns:
        A binarized ranking, where 1 is assigned to
    entries with matching clusters, and 0 otherwise.
    """
    
    rankings = {}
    
    nn.sort_values(by='distance')
    from_item = nn[nn.name==item]
    nn = nn[nn.name!=item]
    
    binarizer = lambda x:1 if x==0 else 0
    
    for c in cluster_names:
        clusters = nn[c].values[:k]
        ground_truth = int(from_item[c])
        
        continuous = [( abs(ground_truth - c) / ground_truth )
                        for c in list(clusters)]
        
        binarized = [binarizer(ground_truth-c) for c in clusters]
        
        rankings[c] = binarized if binarize else continuous
    
    return rankings


def evaluate(k,metadata_df
             ,names=None
             ,cluster_names=['kmeans++'
                 ,'agglomerative_l','agglomerative_m'
                 ,'agglomerative_s','agglomerative_t']
             ,metric='ndcg'
             ,stop_at=50
             ,binary=True):
    
    rankings = []

    for n in names[:stop_at]:
        item, nn = meta_nearest_neighbours(metadata_df,from_name=n,names=names)
        rankings += [(n,nn[nn.name==item],nn,rank_scores(item,nn,binarize=binary,k=k))]
    
    rows = []
    count = 0

    for n,item,nn,cluster_rankings in rankings:
        metrics = []
        
        for c in cluster_rankings.keys():
            ranking = cluster_rankings[c]
            cardinality = nn[nn[c]==item[c].values[0]].shape[0]
            metrics += [ndcg(ranking,cardinality)]
        
        metrics = [n] + metrics
        rows += [tuple(metrics)]
        
    metric_df = pd.DataFrame(rows,columns=['name'] + cluster_names)
    return metric_df


def results(names,metadata_df,
            agglom_labels,kplus_labels,
            cluster_names=['kmeans++'
                 ,'agglomerative_l'
                 ,'agglomerative_m'
                 ,'agglomerative_s'
                 ,'agglomerative_t'],
            top_k=20,
            stop_at=5):

    b_results = []

    for i in range(1,top_k):
        print('{} of {} iterations complete...'.format(i-1,top_k))
        binary_df = evaluate(i,metadata_df,
                            names=list(names),
                            stop_at=stop_at,
                            cluster_names=cluster_names
                            ).drop('name',axis=1)

        b_results += [binary_df.apply(np.mean,axis=0).values]
        header = binary_df.apply(np.mean,axis=0).index

    binary_results = pd.DataFrame(b_results,columns=header)

    return [('binary',binary_results)]


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
