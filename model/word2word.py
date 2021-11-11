from tqdm import tqdm
from transformers import BertTokenizerFast, BartForConditionalGeneration
import time
import torch

# import os
# os.environ['TRANSFORMERS_CACHE'] = '/storage/cache'
# os.environ['TMPDIR/TEMP/TMP'] = '/storage/temp/'
# os.environ['HF_DATASETS_CACHE'] = "/storage/cache"
# from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
import psutil

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
EPOCHS = 5
MAX_LEN = 512
LOAD_SIZE = 500000
print(torch.cuda.get_device_name(0))
torch.cuda.empty_cache()


def dataLoader(tokenizer, data, batch_size):
	query = []
	n = len(data['abbreviated'])
	for i in range(n):
		query.append(data['abbreviated'][i] + " " + data['abbreviations'][i])

	questions_encoding = tokenizer.batch_encode_plus(
		query, padding= True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
	with tokenizer.as_target_tokenizer():
		answers_encoding = tokenizer.batch_encode_plus(
			data['answers'].tolist(), padding= True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

	dataset = TensorDataset(
		questions_encoding['input_ids'], questions_encoding['attention_mask'], answers_encoding["input_ids"])

	return DataLoader(
		dataset,
		batch_size=batch_size
	)


def vramUsage():
	r = torch.cuda.memory_reserved(0)
	a = torch.cuda.memory_allocated(0)
	f = r-a  # free inside reserved
	return f


def evaluate(model, valid):

	model.eval()

	predictions, true = [], []
	with torch.no_grad():
		for batch in valid:

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

	return accuracy_score(true, predictions)




# can be replaced with other ConditionalGeneration model here
print('loading model')
tokenizer = BertTokenizerFast.from_pretrained("fnlp/bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
# model = torch.nn.DataParallel(model)
model = model.to(device)
print('model loaded')
print('memory used:', psutil.virtual_memory()[2])


data = pd.read_csv(
	"/itet-stor/zhejiang/net_scratch/datasetv3/train.csv", index_col='Id').iloc[:LOAD_SIZE]
trainLoader = dataLoader(
	tokenizer,  data,  BATCH_SIZE)

del data

valid_data = pd.read_csv(
	"/itet-stor/zhejiang/net_scratch/datasetv3/valid.csv", index_col='Id').iloc[:2000]
validLoader = dataLoader(
	tokenizer,  valid_data, BATCH_SIZE)

del valid_data

print('All data loaded')
print('memory used:', psutil.virtual_memory()[2])

optimizer = AdamW(
	model.parameters(),
	lr=2e-5,  # recommand:5e-5, 3e-5, 2e-5
	correct_bias=False,
)
scheduler = get_linear_schedule_with_warmup(
	optimizer,
	num_warmup_steps=0,
	num_training_steps=len(trainLoader)*EPOCHS
)
print('optimizer loaded')


print("VRAM left", vramUsage())

print("start training")
best_score = 0
for i in range(EPOCHS):
	model.train()
	t_loss = 0
	start_time = time.time()


	for batch in tqdm(trainLoader):
		model.zero_grad()
		inputs = {
			'input_ids': batch[0].to(device),
			'attention_mask': batch[1].to(device),
			'labels': batch[2].to(device)
		}
		outputs = model(**inputs)
		loss = outputs[0]
		loss.mean().backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		t_loss += loss.mean()
		optimizer.step()
		scheduler.step()
		del inputs
		del outputs
		del loss

		torch.cuda.empty_cache()
		

	print("--- time taken: %s seconds ---" % (time.time() - start_time))
 	print('memory used:', psutil.virtual_memory()[2])
	print("epoch", i)
	print('loss', t_loss)
	score = evaluate(model, validLoader)
	print('accuracy: ', score)
	print("------------------------------")
	if(score>best_score):
		best_score = score  
		torch.save(model.state_dict(),
				'/itet-stor/zhejiang/net_scratch/models/seq2seq_v2_512_500k.model')
  


# sstat --jobs=<JOBID> --format=JobID,AveVMSize%15,MaxRSS%15,AveCPU%15
# sacct --user ${USER} --starttime=2021-10-16 --format=JobID,Start%20,Partition%20,ReqTRES%50,AveVMSize%15,MaxRSS%15,AveCPU%15
