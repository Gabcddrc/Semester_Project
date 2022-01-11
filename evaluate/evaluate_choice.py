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
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
MAX_LEN = 512

def evaluate(model, valid):

	model.eval()

	predictions, true = [], []
	with torch.no_grad():
		for batch in valid:

			inputs = {'input_ids':      batch[0].to(device),
					  'attention_mask': batch[1].to(device),
					  }
			true_label = batch[2].to(device)

			outputs = model(**inputs)
			_, prediction = torch.max(outputs.logits, dim=1)

			predictions.append(prediction.detach().cpu().numpy())
			true.append(true_label.cpu().numpy())

			del inputs
			del true_label
			del prediction
			torch.cuda.empty_cache()

	predictions = np.concatenate(predictions, axis=0)
	true = np.concatenate(true, axis=0)
	return accuracy_score(true,predictions)

def dataLoader(tokenizer, data, batch_size):

	n = len(data)
	ids = []
	masks = []

	for i in range(n):
		prompt = [data['abbreviated'][i],data['abbreviated'][i],data['abbreviated'][i],data['abbreviated'][i]] 
		choice = [data['choice0'][i], data['choice1'][i],data['choice2'][i], data['choice3'][i] ]
		encoding = tokenizer(
		prompt, choice, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
		ids.append(encoding['input_ids'])
		masks.append(encoding['attention_mask'])

	ids = torch.stack(ids)
	masks = torch.stack(masks)

	labels = torch.tensor(data['answer'])

	dataset = TensorDataset(
		ids, masks, labels)


	return DataLoader(
		dataset,
		batch_size=batch_size
	)


print('loading model')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = BertForMultipleChoice.from_pretrained("bert-base-chinese")
# model = torch.nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load("/itet-stor/zhejiang/net_scratch/models/choice_fin.model"))


test_data = pd.read_csv(
	"/itet-stor/zhejiang/net_scratch/dataset_choice/test.csv", index_col=0)

testLoader = dataLoader(
	tokenizer,  test_data, BATCH_SIZE)

del test_data

print("evaluating")


print('accuracy: ', evaluate(model, testLoader))