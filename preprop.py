import pandas as pd 

train_dataset = pd.read_csv('./train.csv', sep=',', names=['text','label','language'])

bigdt = []

for index, row in train_dataset.iterrows():

    tot_text = row['text']
    new_text = []

    for t in tot_text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)

    filtered = " ".join(new_text)

    print (tot_text)
    print (filtered)
    print ('--'*20)        

    bigdt.append(filtered)


train_dataset['filtered'] = bigdt


train_dataset.to_csv('new.csv',index=False)

