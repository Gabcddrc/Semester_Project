from tqdm import tqdm
from transformers import   BertTokenizerFast, BertForMultipleChoice
import re
import torch
# from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import string

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
MAX_LEN = 512
skip = ['a', 'e' ,'i' ,'o','u']
def load_dict(name ):
    with open('/itet-stor/zhejiang/net_scratch/dict/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
def translate(dict, model, text, abb):
    possible_words = dict[abb]
    n = len(possible_words)
    
    prompt = [text, text, text, text]
            
    while n>4:
        selected = []
        
        for i in range(0,n,4):
            choice = []
            choice.append(possible_words[i])
            if i+1 < n:
                choice.append(possible_words[i+1])
            else:
                choice.append(possible_words[i])
            if i+2 < n:
                choice.append(possible_words[i+2])
            else:
                choice.append(possible_words[i])
            if i+3 < n:
                choice.append(possible_words[i+3])
            else:
                choice.append(possible_words[i])
            inputs = tokenizer(prompt,choice, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
       
            outputs = model(
                **{k: v.unsqueeze(0).to(device) for k,v in inputs.items()} )
            
            _, prediction = torch.max(outputs.logits, dim=1)
            prediction = prediction.detach().cpu().numpy()[0]
            selected.append(choice[prediction])
            
        possible_words = selected
        n = len(possible_words)
            
    while len(possible_words) < 4:
        possible_words.append(possible_words[0])
                    

    inputs = tokenizer(prompt,possible_words, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
    outputs = model(
        **{k: v.unsqueeze(0).to(device) for k,v in inputs.items()} )
    
    _, prediction = torch.max(outputs.logits, dim=1)
    i = prediction.detach().cpu().numpy()[0]

    return possible_words[i]



print('loading model')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = BertForMultipleChoice.from_pretrained("bert-base-chinese")
# model = torch.nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load("/itet-stor/zhejiang/net_scratch/models/choice_fin.model"))

weibo = load_dict('weibo')


letters = list(string.ascii_lowercase)
dict = load_dict('dict')

answer = {'abb':[],'text':[],'translation':[]}
model.eval()

for i in tqdm(letters):
    
    if i in skip:
        continue
    
    for j in letters:
        abb = i + j
        if dict.get(abb):
            texts = weibo[abb]
 
            for t in texts:
                if not abb in t:
                    continue
                answer['text'].append(t)
                answer['abb'].append(abb)
                translation = translate(dict, model, t, abb)
                answer['translation'].append(translation)
                
df = pd.DataFrame(answer)
print("saving result")      
df.to_csv('/itet-stor/zhejiang/net_scratch/dataset_choice/answer_dict.csv')
 