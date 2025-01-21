
import os
import typer
from typing_extensions import Annotated
from tqdm import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle

from tools.eagle_standalone import EagleStandalone



app = typer.Typer(pretty_exceptions_show_locals=False)


@torch.inference_mode()
def test_question (tokenizer, model, question, skip_user=True):
	conv = Conversation(
		name="llama3",
		system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
		#system_message=questions[0]['messages'][0]['content'],
		roles=("user", "assistant"),
		sep_style=SeparatorStyle.LLAMA3,
		sep="",
		stop_str="<|eot_id|>",
		stop_token_ids=[128001, 128009],
		messages=[],
	)

	n_prefix = 0
	n_total = 0
	n_accepted = 0

	for msg in question['messages']:
		if msg['role'] == 'system':
			conv.set_system_message(msg['content'])
		else:
			conv.append_message(msg['role'], msg['content'])
			prompt = conv.get_prompt()
			tokens = tokenizer.encode(prompt, return_tensors='pt')

			if tokens.shape[-1] > 4096:
				break
			#print(f'{tokens.shape[-1]=}')

			if skip_user and msg['role'] == 'user':
				n_prefix = tokens.shape[-1]
			else:
				if 'input' in msg and msg['input']:
					continue

				n_prefix = max(0, n_prefix - 1)
				logits = model(tokens.cuda()).logits[0, n_prefix:-2].cpu()
				n_total += logits.shape[0]

				acception = logits.argmax(dim=-1) == tokens[0, n_prefix + 1:-1]
				n_accepted += acception.sum().item()

	return n_total, n_accepted


@torch.inference_mode()
def test_instruction (tokenizer, model, entry, **_):
	prompt = entry['instruction']
	output = entry['response']

	if prompt is None or output is None:
		return 0, 0

	input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
	output_ids = tokenizer.encode(output, return_tensors='pt', add_special_tokens=False)
	ids = torch.cat([input_ids, output_ids], dim=-1)
	#print(f'{ids.shape[1]=}, {input_ids.shape[1]=}')

	if input_ids.shape[1] > 16383:
		return 0, 0

	ids = ids[:, :16384]

	n_prefix = input_ids.shape[-1]
	logits = model(ids.cuda()).logits[0, n_prefix - 1:-1].cpu()
	n_total = logits.shape[0]

	acception = logits.argmax(dim=-1) == ids[0, n_prefix:]
	n_accepted = acception.sum().item()

	return n_total, n_accepted


def eval_dataset (tokenizer, model, data_path, range_str=None, test_user=False, by_instruction=False):
	print(f'Evaluating for dataset {data_path}:')
	questions = datasets.Dataset.from_json(data_path)

	if range_str:
		# convert str to a range, e.g. -100 -> range(100)  5-10 -> range(5, 10)  200- -> range(200, len(questions))
		bi, ei = range_str.split('-')
		indices = range(int(bi or 0), len(questions) if ei == '' else int(ei))
		questions = questions.select(indices)

	test_func = test_instruction if by_instruction else test_question

	n_tokens, n_accepted_tokens = 0, 0
	with tqdm(questions) as progress_bar:
		for question in progress_bar:
			t, a = test_func(tokenizer, model, question, skip_user=not test_user)
			n_tokens += t
			n_accepted_tokens += a

			if n_tokens > 0:
				progress_bar.set_description(f'{n_accepted_tokens}/{n_tokens} = {n_accepted_tokens/n_tokens:.4f}')

	accept_rate = n_accepted_tokens / max(1, n_tokens)
	print(f'	{accept_rate=}, ({n_tokens}/{n_accepted_tokens})')

	return accept_rate



@app.command()
def run_eval (
	model_path: Annotated[str, typer.Option('--model-path', help='Path to the base model.')],
	data_paths: Annotated[str, typer.Option('--data-path', help='Path to the data.')],
	range_str: Annotated[str, typer.Option('--range')]='',
	base_model_path: Annotated[str, typer.Option('--base-model-path', help='Path to the base model for EAGLE.')]=None,
	test_user: Annotated[bool, typer.Option('--test-user')]=False,
	by_instruction: Annotated[bool, typer.Option('--instruction')]=False,
):
	data_paths = data_paths.split(',')

	if 'eagle' in model_path.lower():
		model = EagleStandalone.from_pretrained(base_model_path=base_model_path, ea_model_path=model_path)
		tokenizer = model.tokenizer
	else:
		tokenizer = AutoTokenizer.from_pretrained(model_path)
		model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cuda')

	accept_rates = []
	for data_path in data_paths:
		accept_rates.append(eval_dataset(tokenizer, model, data_path, range_str, test_user, by_instruction=by_instruction))

	set_names = [os.path.basename(data_path).split('.')[0] for data_path in data_paths]

	# print markdown table for results, in column of set_name and accept_rate
	print('| Dataset | Accept Rate |')
	print('| --- | --- |')
	for set_name, accept_rate in zip(set_names, accept_rates):
		print(f'| {set_name} | {accept_rate:.4f} |')


if __name__ == '__main__':
	app()
