from tqdm import tqdm
from transformers import   BertTokenizerFast, BertForMultipleChoice
import time
import torch
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
LOAD_SIZE = 300000
print(torch.cuda.get_device_name(0))
torch.cuda.empty_cache()

#process dataset
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



def main():
	# can be replaced with other ConditionalGeneration model here
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
	model = BertForMultipleChoice.from_pretrained("bert-base-chinese")
	# model = torch.nn.DataParallel(model)
	model = model.to(device)
	print('model loaded')
	print('memory used:', psutil.virtual_memory()[2])

	#load datasets
	data = pd.read_csv(
		"/itet-stor/zhejiang/net_scratch/dataset_choice/train.csv", index_col=0).iloc[:LOAD_SIZE]


	trainLoader = dataLoader(
		tokenizer,  data, BATCH_SIZE)

	del data

	valid_data = pd.read_csv(
		"/itet-stor/zhejiang/net_scratch/dataset_choice/valid.csv", index_col=0).iloc[:3000]
	valid_data.reset_index(drop=True, inplace=True)
	validLoader = dataLoader(
		tokenizer,  valid_data, BATCH_SIZE)

	del valid_data

	print('All data loaded')
	print('memory used:', psutil.virtual_memory()[2])
 
	checkpoint = torch.load('/itet-stor/zhejiang/net_scratch/models/checkpoint.pth')

	optimizer = AdamW(
		model.parameters(),
		lr=2e-5,  # recommand:5e-5, 3e-5, 2e-5
		correct_bias=False,
	)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps= len(trainLoader)*EPOCHS
	)
 
	optimizer.load_state_dict(checkpoint['optimizer'])
	scheduler.load_state_dict(checkpoint['lr_sched'])
	model.load_state_dict(checkpoint['model'])
 
	loss_fn = nn.CrossEntropyLoss().to(device)
	print('optimizer loaded')

 
	print("VRAM left", vramUsage())

	print("start training")
	best_score = 0.9567

	for i in range(checkpoint['epoch'], EPOCHS):

		model.train()
		t_loss = 0
		start_time = time.time()


		for batch in tqdm(trainLoader):

			model.zero_grad()
			inputs = {
				'input_ids': batch[0].to(device),
				'attention_mask': batch[1].to(device)
			}

			true_label = batch[2].to(device)

			outputs = model(**inputs)
			loss = loss_fn(outputs.logits, true_label)
			t_loss += loss.item()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()

			del inputs
			del outputs
			del loss

			torch.cuda.empty_cache()
			

		print("--- time taken: %s seconds ---" % (time.time() - start_time))
		print("epoch", i)
		print('loss', t_loss)
		score = evaluate(model, validLoader)
		print('accuracy: ', score)
		print("------------------------------")
		if(score>best_score):
			best_score = score
			torch.save(model.state_dict(),
				'/itet-stor/zhejiang/net_scratch/models/choice_fin.model')
   
		states = {
			'epoch': i + 1,
			'model': model.state_dict(),
			'lr_sched': scheduler.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		torch.save(states, '/itet-stor/zhejiang/net_scratch/models/checkpoint.pth')

if '__main__' == __name__:
	main()

# sstat --jobs=<JOBID> --format=JobID,AveVMSize%15,MaxRSS%15,AveCPU%15
# sacct --user ${USER} --starttime=2021-10-16 --format=JobID,Start%20,Partition%20,ReqTRES%50,AveVMSize%15,MaxRSS%15,AveCPU%15
