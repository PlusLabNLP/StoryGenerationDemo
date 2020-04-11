import sys
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
count = 0
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


tokenizer = GPT2Tokenizer.from_pretrained('./storyline-cp')
model = GPT2LMHeadModel.from_pretrained('./storyline-cp')
model.eval()


fileflag = sys.argv[1]
if fileflag == "True":
	a = []
	for line in open(sys.argv[2]):
		a.append(line.split(' ====== ')[0]+' ====== ')
	f = open('generated-storyline.txt','w')
	for elem in a:
		input_ids = torch.tensor(tokenizer.encode(elem, add_special_tokens=True)).unsqueeze(0)
		sample_output = model.generate(input_ids,do_sample=True,max_length=50,top_k=20,temperature=0.7,no_repeat_ngram_size=2)
		f.write(tokenizer.decode(sample_output[0], skip_special_tokens=True)+'\n')
		if count>10:
			break
		count = count+1
else:
	elem = sys.argv[2]
	input_ids = torch.tensor(tokenizer.encode(elem, add_special_tokens=True)).unsqueeze(0)
	sample_output = model.generate(input_ids,do_sample=True,max_length=50,top_k=20,temperature=0.7,no_repeat_ngram_size=2)
	print(tokenizer.decode(sample_output[0], skip_special_tokens=True)+'\n')
