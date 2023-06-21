from transformers import GPT2Tokenizer,GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("congcongwang/distilgpt2_fine_tuned_coder")
model = GPT2LMHeadModel.from_pretrained("congcongwang/distilgpt2_fine_tuned_coder")

def generate_next_token(batch, use_cuda=True):
	if use_cuda:
		model.to("cuda")

	batch = ["<java> " + x for x in batch]
	input_ids = tokenizer(batch, padding=True, return_tensors='pt').to('cuda' if use_cuda else 'cpu')
	outputs = model.generate(
		**input_ids,
		max_length=128,
		temperature=0.7,
		num_return_sequences=1,
		pad_token_id=tokenizer.eos_token_id
	)
	decoded = [x.replace('\n', '') for x in tokenizer.batch_decode(outputs, skip_special_tokens=True)]

	try:
		res = [extract_next_token(dec, context.replace('<java> ', '')) for dec, context in zip(decoded, batch)]
		if len(res) == len(batch):
			return res
		return ['this']*len(batch)
	except:
		return ['this']*len(batch)

def extract_next_token(decoded, context):
	i = len(''.join(context.split()))
	new_str = ''
	for ch in decoded:
		new_str += ch
		if ch != ' ':
			i -= 1
		if i <= 0:
			break
	try:
		return decoded.split(new_str)[1].split()[0]
	except:
		return 'this'
