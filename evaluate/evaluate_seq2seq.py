from tqdm import tqdm
from transformers import BertTokenizerFast, BartForConditionalGeneration
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

def predict(model, valid, tokenizer):

	model.eval()

	predictions, true = [], []
	with torch.no_grad():
		for batch in tqdm(valid):

			inputs = {'input_ids':      batch[0].to(device),
					  'attention_mask': batch[1].to(device),
					  }
			true_label = batch[2].to(device)
			prediction = model.generate(
				input_ids=inputs['input_ids'], max_length=MAX_LEN, early_stopping=True)

			predictions += tokenizer.batch_decode(
				prediction, skip_special_tokens=True)
				
			true += tokenizer.batch_decode(true_label,
										   skip_special_tokens=True)
			del inputs
			del prediction
			torch.cuda.empty_cache()

	return predictions, true

def scoring(prediction, true):

	correct = 0
	mistake_sentence = []
	mistake_prediction = []
	n = len(prediction)
	
	if n != len(true):
		print('size mismatch')
	for i in range(n):

		if prediction[i] != true[i]:
			mistake_sentence.append(true[i])
			mistake_prediction.append(prediction[i])
		else:
			correct += 1

	df = pd.DataFrame({'target':mistake_sentence, 'prediction': mistake_prediction})
	df.to_csv('/itet-stor/zhejiang/net_scratch/datasetv2/mistakes/seq2seq_512_mistakes')
	return correct/n
	

def dataLoader(tokenizer, X, y, batch_size):
	questions_encoding = tokenizer.batch_encode_plus(
		X.tolist(), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
	with tokenizer.as_target_tokenizer():
		answers_encoding = tokenizer.batch_encode_plus(
			y.tolist(), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

	dataset = TensorDataset(
		questions_encoding['input_ids'], questions_encoding['attention_mask'], answers_encoding["input_ids"])
	return DataLoader(
		dataset,
		batch_size=batch_size
	)


print('loading model')
tokenizer = BertTokenizerFast.from_pretrained("fnlp/bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
# model = torch.nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load("/itet-stor/zhejiang/net_scratch/models/seq2seq_512.model"))


test_data = pd.read_csv(
	"/itet-stor/zhejiang/net_scratch/datasetv2/test.csv", index_col='Id')

testLoader = dataLoader(
	tokenizer,  test_data['abbreviated'],  test_data['sentences'], BATCH_SIZE)

del test_data

print("evaluating")
prediction, true = predict(model, testLoader, tokenizer)


print('accuracy: ', scoring(prediction, true))