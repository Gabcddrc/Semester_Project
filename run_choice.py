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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
MAX_LEN = 512



def main():
    print('loading model')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = BertForMultipleChoice.from_pretrained("bert-base-chinese")
    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(
        '/itet-stor/zhejiang/net_scratch/models/choice_fin.model'))
    model.eval()
    while True:
        
        text = input("Enter Text:")
        choice0 = input("Enter choice 0:")
        choice1 = input("Enter choice 1:")
        choice2 = input("Enter choice 2:")
        choice3 = input("Enter choice 3:")
        
        choice = [choice0, choice1, choice2, choice3]
        prompt = [text, text, text, text]

        
        if text == "quit":
            break
        inputs = tokenizer(prompt,choice, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

        outputs = model(
            **{k: v.unsqueeze(0).to(device) for k,v in inputs.items()} )
        
        _, prediction = torch.max(outputs.logits, dim=1)
        i = prediction.detach().cpu().numpy()[0]
  
        print(choice[i])
        
        
        # print("".join(translation[0].split(" ")))


if '__main__' == __name__:
    main()