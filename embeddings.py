from bert_embedding import BertEmbedding
from bert.utils import zomatoPreprocess  

import numpy as np 
import pandas as pd
  
import warnings
warnings.filterwarnings('ignore')


def stack(embeddings,stacked=[],n=0):

    if len(stacked)>0: x = stacked
    else: x = embeddings[0]
    
    try: stacked = np.add(np.array(x), embeddings[1])
    except: return np.array(x)
    
    n += 1
    
    return stack(embeddings[1:],stacked=stacked,n=n)




def meanVec(embeddings):
    
    set_of_means = []
    dimensionality = embeddings[0].shape[0]
    
    n = len(embeddings)
    mean_vector = []

    for dim in range(dimensionality):
        cumulative_dimension = 0

        for i in range(n):
            try:
                cumulative_dimension += embeddings[i][dim]
            except:
                print(i,dim)

        mean_vector += [cumulative_dimension/n]
    
    return np.array(mean_vector)




def __bagofBERTs(restaurants,average,urls,berts):
    
    output = []
    x = list(zip(restaurants,urls))
    for (restaurant,url),(cuisines,embeddings) in zip(x,berts):
        if not(average): 
            embedding = stack(embeddings)
        else:
            embedding = meanVec(embeddings)
            
        output += [(restaurant,url,cuisines,embedding)]
        
    return output


def getEmbeddings(n_restaurants=None,average=True,city='sydney',cuisines=True):
    
    print('Cleaning Zomato data for {}.\n'.format(city))
    cuisines, urls, names = zomatoPreprocess(city.lower(),cuisines=cuisines)
    
    if n_restaurants == None: n_restaurants = len(cuisines)
    
    print('Retrieving BERT sentence representations for {} restuarants...\n'.format(n_restaurants))
    __bert_embedding = BertEmbedding(model='bert_12_768_12')
    __berts = __bert_embedding(cuisines[:n_restaurants])
    bagofembeddings = __bagofBERTs(names, average, urls, __berts)
    
    print('Complete.')
    
    filtrd = [(n,u,c,e) for n,u,c,e
              in bagofembeddings 
                  if len(e.shape)>0]

    cuisines = [c for n,u,c,e in filtrd]
    embeds = [e for n,u,c,e in filtrd]
    names = [n for n,u,c,e in filtrd]
    urls = [u for n,u,c,e in filtrd]
    
    return names,cuisines,urls,embeds,bagofembeddings