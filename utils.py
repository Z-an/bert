import pandas as pd

def zomatoPreprocess(city,cuisines=True):

    zomato = pd.read_csv(city+'.csv')
    
    if cuisines:
        
        zomato = zomato[['name','url','cuisines']].dropna(axis=0)

        sentences = zomato.cuisines.apply(lambda x:x.replace('[','')
                                                    .replace(']','')
                                                    .replace('"','')
                                                    .replace(',',' ')
                                                    .replace('and','') + ' food'
                                                    .replace('  ',' ')).values
    else:
        
        zomato = zomato[['name','url','food_items']].dropna(axis=0)
    
        food = zomato.food_items.apply(lambda x:[food.rstrip() 
                                             for food 
                                             in (str(x).replace('\\n','').split(','))])
        new_f = ''
        new_food = []
        for f in food:
            for i in range(len(f)):
                if i == len(f)-1: new_f += f[i]
                else: new_f += f[i]+'. '
            new_food += [new_f]
            new_f = ''
            
        sentences = new_food

    names = zomato.name.values
    urls = zomato.url.apply(lambda x:x.replace('\n','')).values
    
    return sentences, urls, names