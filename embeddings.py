from bert_embedding import BertEmbedding
from utils import zomatoPreprocess  

import numpy as np 
import pandas as pd
  
import warnings
warnings.filterwarnings('ignore')


def stack(embeddings,stacked=[],n=0):

    if len(stacked)>0: x = stacked
    else: x = embeddings[0]
    
    try:stacked = np.add(np.array(x), embeddings[1])
    except:return np.array(x)
    
    n += 1
    
    return stack(embeddings[1:],stacked=stacked,n=n)


def __bagofBERTs(restaurants,urls,berts):
    
    output = []
    x = list(zip(restaurants,urls))
    for (restaurant,url),(cuisines,embeddings) in zip(x,berts):
        embedding = stack(embeddings)
        output += [(restaurant,url,cuisines,embedding)]
        
    return output


def getEmbeddings(n_restaurants=None,city='sydney',cuisines=True):
    
    print('Cleaning Zomato data for {}.\n'.format(city))
    cuisines, urls, names = zomatoPreprocess(city.lower(),cuisines=cuisines)
    
    if n_restaurants == None: n_restaurants = len(cuisines)
    
    print('Retrieving BERT sentence representations for {} restuarants...\n'.format(n_restaurants))
    __bert_embedding = BertEmbedding(model='bert_12_768_12')
    __berts = __bert_embedding(cuisines[:n_restaurants])
    bagofembeddings = __bagofBERTs(names, urls, __berts)
    
    print('Complete.')
    
    filtrd = [(n,u,c,e) for n,u,c,e
              in bagofembeddings 
                  if len(e.shape)>0]

    cuisines = [c for n,u,c,e in filtrd]
    embeds = [e for n,u,c,e in filtrd]
    names = [n for n,u,c,e in filtrd]
    urls = [u for n,u,c,e in filtrd]
    
    return names,cuisines,urls,embeds,bagofembeddings