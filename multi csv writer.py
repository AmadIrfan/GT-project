import pandas as pd

df = pd.read_csv("output.csv", delimiter=";", encoding="latin1")
category = df["category"].to_list()
content = df["content"].to_list()


def to_CSVS():
    idx=1
    print(len(category))
    for i in range(len(category)):
        data = {
            "category": [category[i]],
            "content": [content[i]],
        }
        ds = pd.DataFrame(data)
        ds.to_csv(category[i] +' {0}.csv'.format(idx) , index=False)
        print( category[idx] +' {0}.csv '.format(idx) + 'created successfully.')
        
        if(idx==15):
            idx=1
        else:    
            idx+=1



to_CSVS()