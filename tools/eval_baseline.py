
import typer
from typing_extensions import Annotated
import os
from accelerate.utils import set_seed
from tqdm import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import shortuuid


set_seed(0)

app = typer.Typer()


@torch.inference_mode()
def get_model_answers(
	base_model_path,
	model_id,
	questions,
	answer_file,
	max_new_token,
	num_choices,
):
	#print('questions:', questions)
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

	model = AutoModelForCausalLM.from_pretrained(
		base_model_path,
		torch_dtype=torch.float16,
		device_map='cuda',
	)
	model.eval()

	pad_token_id = tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("<|end_of_text|>")
	eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.eos_token_id is None or tokenizer.eos_token_id == pad_token_id else tokenizer.eos_token_id

	# warmup
	question0 = questions[0]
	for _ in tqdm(range(3), desc='Warming up'):
		torch.manual_seed(0)

		messages = []
		for j in range(len(question0["turns"])):
			qs = question0["turns"][j]
			messages.append({
				"role": "user",
				"content": qs,
			})
			prompt = tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
			)
			input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

			# try:
			torch.cuda.synchronize()

			output_ids = model.generate(
				torch.as_tensor(input_ids).cuda(),
				do_sample=False,
				temperature=None,
				top_p=None,
				top_k=None,
				eos_token_id=eos_token_id,
				pad_token_id=pad_token_id,
				max_new_tokens=1,
			)
			torch.cuda.synchronize()
			output_ids = output_ids[0][len(input_ids[0]):]
			# be consistent with the template's stop_token_ids
			stop_token_ids = [
				tokenizer.eos_token_id,
				tokenizer.convert_tokens_to_ids("<|eot_id|>")
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
	#print('Warmup done.')

	for question in tqdm(questions):
		choices = []
		for i in range(num_choices):
			torch.manual_seed(i)
			messages = []
			turns = []
			idxs = []
			new_tokens = []
			wall_time = []
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
				if len(input_ids[0]) > 4096:
					break

				torch.cuda.synchronize()
				start_time = time.time()

				output_ids = model.generate(
					torch.as_tensor(input_ids).cuda(),
					do_sample=False,
					temperature=None,
					top_p=None,
					top_k=None,
					eos_token_id=eos_token_id,
					pad_token_id=pad_token_id,
					max_new_tokens=max_new_token,
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
				#idxs.append(int(idx))
				new_tokens.append(int(new_token))
				wall_time.append(total_time)
				messages.append({
					"role": "assistant",
					"content": output
				})
			# torch.cuda.empty_cache()
			choices.append({"index": i, "turns": turns, "new_tokens": new_tokens, "wall_time": wall_time})

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
	data_path: Annotated[str, typer.Option('--data-path', help='Path to the data.')],
	max_new_token: Annotated[int, typer.Option('--max-new-token', help='Maximum number of new tokens to generate.')] = 512,
	#num_choices: Annotated[int, typer.Option('--num-choices', help='Number of choices.')] = 1,
	#num_gpus_per_model,
	#num_gpus_total,
):
	model_id = base_model_path.split('/')[-1]

	data_name = os.path.basename(data_path)
	data_name = os.path.splitext(data_name)[0]

	answer_file = f'./evaluation/{model_id}-{data_name}-baseline.jsonl'

	questions = datasets.Dataset.from_json(data_path)

	if os.path.exists(answer_file):
		answers = datasets.Dataset.from_json(answer_file)
		ids = answers['question_id']
		questions = questions.filter(lambda x: x['question_id'] not in ids)
		print(f'skipped {len(ids)} finished questions')

	ans_handles = []
	ans_handles.append(
		get_model_answers(
			base_model_path,
			model_id,
			questions,
			answer_file,
			max_new_token,
			1,
		)
	)


if __name__ == '__main__':
	app()
