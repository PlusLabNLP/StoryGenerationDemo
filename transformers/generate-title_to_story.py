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



fileflag = sys.argv[1]
if fileflag == "True":
	a = []
	tokenizer = GPT2Tokenizer.from_pretrained('./storyline-cp')
	model = GPT2LMHeadModel.from_pretrained('./storyline-cp')
	model.eval()

	tokenizer1 = GPT2Tokenizer.from_pretrained('./story-cp')
	model1 = GPT2LMHeadModel.from_pretrained('./story-cp')
	model1.eval()
	for line in open(sys.argv[2]):
		a.append(line.split('  ======  ')[0]+'  ======  ')
	f = open('generated-story.txt','w')
	for elem in a:
		elem = elem.replace(' ======  ','')
		input_ids = torch.tensor(tokenizer.encode(elem, add_special_tokens=True)).unsqueeze(0)
		sample_output = model.generate(input_ids,do_sample=True,max_length=150,top_k=20,temperature=0.7,no_repeat_ngram_size=3)
		op = tokenizer.decode(sample_output[0], skip_special_tokens=True)
			
		input_ids = torch.tensor(tokenizer1.encode(op, add_special_tokens=True)).unsqueeze(0)
		sample_output = model1.generate(input_ids,do_sample=True,max_length=150,top_k=20,temperature=0.7,no_repeat_ngram_size=3)
		story = tokenizer1.decode(sample_output[0], skip_special_tokens=True)
		story = story.split(' $ ')
		story = [o.capitalize() for o in story]
		story = '. $ '.join(story)
		f.write(story+'\n')


else:
	elem = sys.argv[2]
	input_ids = torch.tensor(tokenizer.encode(elem, add_special_tokens=True)).unsqueeze(0)
	sample_output = model.generate(input_ids,do_sample=True,max_length=150,top_k=20,temperature=0.7,no_repeat_ngram_size=3)
	op = tokenizer.decode(sample_output[0], skip_special_tokens=True)
			
	input_ids = torch.tensor(tokenizer1.encode(op, add_special_tokens=True)).unsqueeze(0)
	sample_output = model1.generate(input_ids,do_sample=True,max_length=150,top_k=20,temperature=0.7,no_repeat_ngram_size=3)
	story = tokenizer1.decode(sample_output[0], skip_special_tokens=True)
	story = story.split(' $ ')
	story = [o.capitalize() for o in story]
	story = '. $ '.join(story)
	print(story)
