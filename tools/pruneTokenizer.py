
import os
import typer
from typing_extensions import Annotated
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm



app = typer.Typer(pretty_exceptions_show_locals=False)


def resort_tokens(new_pairs, n_reserve):
	head = []

	for k, v in tqdm(new_pairs[:n_reserve]):
		done = False
		for h, hi in head:
			if k in h:
				head.append((k, hi))
				done = True
				break

		if not done:
			head.append((k, v))

	tail = []

	for k, v in tqdm(new_pairs[n_reserve:]):
		done = False
		for h, hi in head:
			if k in h:
				tail.append((k, hi))
				done = True
				break

		if not done:
			tail.append((k, v))

	resort_pairs = (head + tail)
	resort_pairs.sort(key=lambda x: x[1])

	in_pairs = [(p[0], i) for i, p in enumerate(resort_pairs[:n_reserve])]

	return in_pairs


@app.command()
def main (
	token_freq: Annotated[str, typer.Option('--token-freq', help='A file of the tensor of token frequencies.')],
	source_model: Annotated[str, typer.Option('--source-model', help='The source tokenizer.')],
	target_model: Annotated[str, typer.Option('--target-model', help='The target tokenizer.')],
	n_reserve: Annotated[int, typer.Option('--n-reserve', help='The number of tokens to reserve.')]
):
	indices = torch.load(token_freq)
	tokenizer_config = json.load(open(os.path.join(source_model, 'tokenizer.json'), 'r'))

	source_vocab_size = len(tokenizer_config['model']['vocab'].keys())

	# shift id values
	for at in tokenizer_config['added_tokens']:
		at['id'] -= source_vocab_size - n_reserve

	if '<|begin_of_text|>' in tokenizer_config['post_processor']['processors'][-1]['special_tokens']:
		bot = tokenizer_config['post_processor']['processors'][-1]['special_tokens']['<|begin_of_text|>']
		bot['ids'] = [id - (source_vocab_size - n_reserve) for id in bot['ids']]

	# prune vocab
	pairs = list(tokenizer_config['model']['vocab'].items())
	new_pairs = [(k, (indices == v).nonzero().item()) for k, v in pairs]

	# optional resort_tokens
	#new_pairs = resort_tokens(new_pairs, n_reserve)

	tokenizer_config['model']['vocab'] = dict(new_pairs)

	# prune merges
	vocab = tokenizer_config['model']['vocab']

	def validate_merge (m):
		t1, t2 = m.split(' ')
		#return t1 in vocab and t2 in vocab and (t1 + t2) in vocab
		return (t1 + t2) in vocab

	new_merges = [m for m in tokenizer_config['model']['merges'] if validate_merge(m)]
	tokenizer_config['model']['merges'] = new_merges

	json.dump(tokenizer_config, open(os.path.join(target_model, 'tokenizer.json'), 'w'), indent=2, ensure_ascii=False)

	print('Target config wrote.')

	tokenizer = AutoTokenizer.from_pretrained(target_model)
	print(f'{tokenizer=}')

	print('Done.')


if __name__ == '__main__':
	app()
