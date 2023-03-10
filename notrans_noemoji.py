import pandas as pd
import emoji
import numpy as np
import gensim.models as gsm
from tqdm import tqdm
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import sys
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
from scipy import stats



LMTokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
LMModel = AutoModel.from_pretrained(sys.argv[1],cache_dir='/scratch')

device = 'cuda' if cuda.is_available() else 'cpu'

train_dataset = pd.read_csv('og_train.csv', sep=',', names=['non_filter','label','language'])
testing_dataset = pd.read_csv('test.csv', sep=',', names=['non_filter','label','language','text'])

MAX_LEN = 256
TRAIN_BATCH_SIZE = int(sys.argv[2]) # 4
VALID_BATCH_SIZE = int(sys.argv[2]) # 4
LEARNING_RATE = float(sys.argv[3])  # 0.00001
drop_out = float(sys.argv[4])       # 0.1
EPOCHS = 3
tokenizer = LMTokenizer

output_file_name = str(sys.argv[5]) + "_" + str(sys.argv[1]).split('/')[0] + "_" + str(sys.argv[2]) + "_" + str(sys.argv[3]) + "_" + str(sys.argv[4]) + ".txt"
output_file_csv = str(sys.argv[5]) + "_" + str(sys.argv[1]).split('/')[0] + "_" + str(sys.argv[2]) + "_" + str(sys.argv[3]) + "_" + str(sys.argv[4]) + ".csv"
file = open(output_file_name,'w')

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len,mode):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode=mode
        self.e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
        
    def __getitem__(self, index):
        tot_text = str(self.data.non_filter[index])
        tot_text = " ".join(tot_text.split())

        tot_text = self.data.language[index].lower() + ' [SEP] ' + tot_text

        emlist  = emoji.emoji_list(tot_text)
        eml = [x['emoji'] for x in emlist]
        biglist = []
        bigstr = ''
        for em in eml:
            bigstr += em
            if em in self.e2v:
                biglist.append(self.e2v[em])
                #print ("nolo")
            else :
                pass
                #print ("lol")

        if biglist == []:
            biglist = [[0]*300]
        finmean = np.mean(biglist,axis=0)
        #print(finmean)
        #print(np.size(finmean))
        
        #tot_text += '[SEP] ' + bigstr

        inputs = self.tokenizer.encode_plus(
            tot_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        tot_text_ids = inputs['input_ids']
        tot_text_mask = inputs['attention_mask']
        if self.mode=='train':
            return {
                'text_ids': torch.tensor(tot_text_ids, dtype=torch.long),
                'text_mask': torch.tensor(tot_text_mask, dtype=torch.long),
                'emoji': finmean,
                'targets': torch.tensor(float(self.data.label[index]), dtype=torch.float32)
            } 
        else :
            return {
                'text_ids': torch.tensor(tot_text_ids, dtype=torch.long),
                'text_mask': torch.tensor(tot_text_mask, dtype=torch.long),
                'emoji': finmean,
            } 

    
    def __len__(self):
        return self.len


training_set = Triage(train_dataset, tokenizer, MAX_LEN,"train")
testing_set = Triage(testing_dataset, tokenizer, MAX_LEN,"test")

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class LMClass(torch.nn.Module):
    def __init__(self):
        super(LMClass, self).__init__()
        self.l1 = LMModel
        self.pre_classifier = torch.nn.Linear(768, 512)
        self.dropout = torch.nn.Dropout(drop_out)
        self.classifier = torch.nn.Linear(512, 1)

    def forward(self, data):
        
        input_ids = data['text_ids'].to(device, dtype = torch.long)
        attention_mask = data['text_mask'].to(device, dtype = torch.long)
        
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state1 = output_1[0]


        #emjvec = data['emoji'].to(device,dtype=torch.float32)

        pooler = hidden_state1[:, 0]
        #pooler = torch.cat((pooler, emjvec), 1)

        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = LMClass()
model.to(device)
#weights = [0.30435841, 1.34843581, 2.91375291, 7.57575758, 1.78062678,1.06837607]
#class_weights = torch.FloatTensor(weights).to(device)
#loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

loss_function =  torch.nn.MSELoss()


optimizer = torch.optim.AdamW(params = model.parameters(), lr=LEARNING_RATE)

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0), total =13081/TRAIN_BATCH_SIZE):
        targets = data['targets'].to(device, dtype = torch.float32)

        outputs = model(data)
        targets = torch.unsqueeze(targets,1)
        #print(outputs.shape)
        #print(targets.shape)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        #print (outputs.data)
        #big_val, big_idx = torch.max(outputs.data, dim=1)
        #n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #file.write(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}\n')
    print(f'The Total loss for Epoch {epoch}: {(tr_loss)/nb_tr_steps}\n')
    epoch_loss = tr_loss/nb_tr_steps
    #epoch_accu = (n_correct*100)/nb_tr_examples
    file.write(f"Training Loss Epoch: {epoch_loss}\n")
    #file.write(f"Training Accuracy Epoch: {epoch_accu}\n")
    file.write("\n")
    return

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; tr_loss = 0
    nb_tr_steps =0
    nb_tr_examples =0
    pred = []
    act = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0),total=13697/TRAIN_BATCH_SIZE):
            outputs = model(data).squeeze()
            #print (outputs.shape)
            #print (outputs.size(0))
            pred += outputs.tolist()
            nb_tr_steps += 1
            nb_tr_examples+=outputs.size(0)
            
    #epoch_loss = tr_loss/nb_tr_steps
    epoch_loss = 0
    #epoch_accu = (n_correct*100)/nb_tr_examples
    file.write(f"Validation Loss Epoch: {epoch_loss}\n")
    #file.write(f"Validation Accuracy Epoch: {epoch_accu}\n")
    #mf1 = f1_score(act, pred, average='macro')
    #file.write(f"Validation Macro F1: {mf1}\n")
    #print (stats.pearsonr( act,pred))
    testdf = pd.read_csv('test.csv', sep=',', names=['non_filter','language','text'])
    testdf.drop(['text'], axis=1, inplace=True)
    testdf['label'] = pred
    print(pred)
    testdf.to_csv(output_file_csv,index=False)
    return epoch_loss, 0

best_mf1 = 0
best_epoch = 0
best_acc = 0

for epoch in range(EPOCHS):
    train(epoch)
    if epoch != 0 :
        epl,pear = valid(model, testing_loader)
        if epl < best_mf1:
            best_mf1 = epl
            #best_acc = acc
            best_epoch = epoch+1
            best_acc = pear

    file.write("\n")



file.write("Best \nAccuracy: {0} \nbest loss: {1}\nAt Epoch: {2}\n".format(best_mf1,best_mf1,best_epoch))
file.close()


