
import typer
from typing_extensions import Annotated
import os
from accelerate.utils import set_seed
from tqdm import tqdm
import torch
import datasets
import time
import json
import shortuuid

from axolotl.models.eagle import EaModel


set_seed(0)

app = typer.Typer(pretty_exceptions_show_locals=False)


def complete_chat (question, tokenizer, model, eos_token_id, pad_token_id, max_new_token, pad_head_zero=False):
	messages = []
	turns = []
	taus = []
	new_tokens = []
	wall_time = []
	profiler = {}

	if question['system'] is not None:
		messages.append(dict(
			role="system",
			content=question['system'],
		))

	for j in range(len(question["turns"])):
		qs = question["turns"][j]
		messages.append({
			"role": "user",
			"content": qs
		})
		prompt = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)
		input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids
		#print('input_ids:', len(input_ids[0]))
		if len(input_ids[0]) > 3400:
			break

		torch.cuda.synchronize()
		start_time = time.time()

		output_ids, new_token, idx, accept_lengths = model.eagenerate(
			torch.as_tensor(input_ids).cuda(),
			temperature=0,
			log=True,
			is_llama3=True,
			max_new_tokens=max_new_token,
			max_length=8192,
			profiler=profiler,
			pad_head_zero=pad_head_zero,
		)
		torch.cuda.synchronize()
		total_time = time.time() - start_time
		output_ids = output_ids[0][len(input_ids[0]):]

		new_token = output_ids.shape[0]

		# be consistent with the template's stop_token_ids
		stop_token_ids = [
			eos_token_id,
			pad_token_id,
		]

		if stop_token_ids:
			stop_token_ids_index = [
				i
				for i, id in enumerate(output_ids)
				if id in stop_token_ids
			]
			if len(stop_token_ids_index) > 0:
				output_ids = output_ids[: stop_token_ids_index[0]]

		output = tokenizer.decode(
			output_ids,
			spaces_between_special_tokens=False,
		)
		for special_token in tokenizer.special_tokens_map.values():
			if isinstance(special_token, list):
				for special_tok in special_token:
					output = output.replace(special_tok, "")
			else:
				output = output.replace(special_token, "")
		output = output.strip()

		turns.append(output)
		taus += accept_lengths
		new_tokens.append(int(new_token))
		wall_time.append(total_time)
		messages.append({
			"role": "assistant",
			"content": output
		})

	return turns, taus, new_tokens, wall_time, profiler


def complete_instruction (question, tokenizer, model, eos_token_id, pad_token_id, max_new_token, pad_head_zero=False):
	turns = []
	turns = []
	taus = []
	new_tokens = []
	wall_time = []
	profiler = {}

	prompt = question['prompt']
	stop_strings = json.loads(question['stop'].replace('\\\\', '\\'))
	stop_ids = [[eos_token_id], [pad_token_id]] + [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

	input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
	input_ids = input_ids[:, -3072:]

	torch.cuda.synchronize()
	start_time = time.time()

	output_ids, new_token, idx, accept_lengths = model.eagenerate(
		input_ids.cuda(),
		temperature=0,
		log=True,
		is_llama3=True,
		max_new_tokens=max_new_token,
		max_length=8192,
		profiler=profiler,
		stop_ids=stop_ids,
		pad_head_zero=pad_head_zero,
	)
	torch.cuda.synchronize()
	total_time = time.time() - start_time
	output_ids = output_ids[0][len(input_ids[0]):]

	new_token = output_ids.shape[0]

	output = tokenizer.decode(
		output_ids,
		spaces_between_special_tokens=False,
	)
	for special_token in tokenizer.special_tokens_map.values():
		if isinstance(special_token, list):
			for special_tok in special_token:
				output = output.replace(special_tok, "")
		else:
			output = output.replace(special_token, "")
	output = output.strip()

	turns.append(output)
	taus += accept_lengths
	new_tokens.append(int(new_token))
	wall_time.append(total_time)

	return turns, taus, new_tokens, wall_time, profiler


@torch.inference_mode()
def get_model_answers(
	base_model_path,
	ea_model_path,
	model_id,
	questions,
	answer_file,
	max_new_token,
	num_choices,
	total_token,
	depth,
	top_k,
	by_instruction,
	pad_head_zero=False,
):
	model = EaModel.from_pretrained(
		base_model_path=base_model_path,
		ea_model_path=ea_model_path,
		total_token=total_token,
		depth=depth,
		top_k=top_k,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		# load_in_8bit=True,
		device_map="auto"
	)
	model.eval()

	tokenizer = model.get_tokenizer()

	pad_token_id = tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("<|end_of_text|>")
	eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.eos_token_id is None or tokenizer.eos_token_id == pad_token_id else tokenizer.eos_token_id

	complete_method = complete_instruction if by_instruction else complete_chat

	# warmup
	question0 = questions[0]
	for _ in tqdm(range(3), desc='Warming up'):
		torch.manual_seed(0)
		complete_method(
			question0,
			tokenizer,
			model,
			eos_token_id,
			pad_token_id,
			max_new_token=16,
		)

	for question in tqdm(questions):
		choices = []
		for i in range(num_choices):
			torch.manual_seed(i)
			turns, taus, new_tokens, wall_time, profiler = complete_method(
				question,
				tokenizer,
				model,
				eos_token_id,
				pad_token_id,
				max_new_token,
				pad_head_zero=pad_head_zero,
			)

			# torch.cuda.empty_cache()
			if not "base" in profiler:
				continue

			choices.append({
				"index": i,
				"turns": turns,
				"taus": taus,
				"new_tokens": new_tokens,
				"wall_time": wall_time,
				"base_time": profiler.get("base", 0),
				"ealayer_time": profiler.get("ea_layer", 0),
				"head_time": profiler.get("head", 0),
			})

		# Dump answers
		os.makedirs(os.path.dirname(answer_file), exist_ok=True)
		with open(os.path.expanduser(answer_file), "a") as fout:
			ans_json = {
				"question_id": question["question_id"],
				"answer_id": shortuuid.uuid(),
				"model_id": model_id,
				"choices": choices,
				"tstamp": time.time(),
			}
			fout.write(json.dumps(ans_json) + "\n")


@app.command()
def run_eval(
	base_model_path: Annotated[str, typer.Option('--base-model-path', help='Path to the base model.')],
	ea_model_path: Annotated[str, typer.Option('--ea-model-path', help='Path to the EAGLE model.')],
	data_path: Annotated[str, typer.Option('--data-path', help='Path to the data.')],
	postfix: Annotated[str, typer.Option('--postfix', help='Postfix on answer file name.')],
	max_new_token: Annotated[int, typer.Option('--max-new-token', help='Maximum number of new tokens to generate.')] = 512,
	total_token: Annotated[int, typer.Option('--total-token', help='Total draft token number.')] = 60,
	depth: Annotated[int, typer.Option('--depth', help='Tree attention depth.')] = 5,
	top_k: Annotated[int, typer.Option('--top-k', help='K of tree attention top K choices.')] = 10,
	by_instruction: Annotated[bool, typer.Option('--instruct', help='Use instruction data against chat.')] = False,
	pad_head_zero: Annotated[bool, typer.Option('--pad-head-zero', help='Add a zero state on the head of KV cache.')] = False,
):
	model_id = base_model_path.split('/')[-1]

	data_name = os.path.basename(data_path)
	data_name = os.path.splitext(data_name)[0].replace('.local', '')

	os.makedirs(f'./evaluation/{model_id}', exist_ok=True)

	answer_file = f'./evaluation/{model_id}/{data_name}-eagle-{postfix}.jsonl'

	questions = datasets.Dataset.from_json(data_path)

	if os.path.exists(answer_file):
		answers = datasets.Dataset.from_json(answer_file)
		ids = answers['question_id']
		questions = questions.filter(lambda x: x['question_id'] not in ids)
		if len(questions) == 0:
			print(f'Already completed: {data_name}.')
			return
		print(f'skipped {len(ids)} finished questions')

	ans_handles = []
	ans_handles.append(
		get_model_answers(
			base_model_path,
			ea_model_path,
			model_id,
			questions,
			answer_file,
			max_new_token,
			1,
			total_token=total_token,
			depth=depth,
			top_k=top_k,
			by_instruction=by_instruction,
			pad_head_zero=pad_head_zero,
		)
	)


if __name__ == '__main__':
	app()
