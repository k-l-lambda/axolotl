
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
from fms.models import get_model
#from fms.utils import tokenizers
from transformers import AutoTokenizer

import axolotl.models.fms_extras.models.paged_llama
from axolotl.models.fms_extras.models.hf.modeling_mlp_speculator import MLPSpeculatorPreTrainedModel
from axolotl.models.fms_extras.utils.cache.paged import PagedKVCacheManager
from axolotl.models.fms_extras.utils.generation import speculative_generate



set_seed(0)
torch.set_default_dtype(torch.half)
torch.set_grad_enabled(False)

app = typer.Typer(pretty_exceptions_show_locals=False)


def complete_chat (question, tokenizer, kv_cache_manager, model, speculator, eos_token_id, pad_token_id,
	max_new_token, pad_head_zero=False, threshes=[4,3,2,2]):
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

		output_ids, n_steps, ttft, generated_token_time_out, accept_lengths = speculative_generate(
			model,
			torch.as_tensor(input_ids).cuda(),
			speculator,
			kv_cache_manager,
			new_tokens=100,
			max_seq_len=model.config.max_expected_seq_len,
			decode_model=None,
			# todo: we can only reduce-overhead for now when batch size is 1
			flattening=True,
			cudagraphs=False,
			threshes=threshes,
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
		#print(f'{output=}')

		turns.append(output)
		taus += accept_lengths
		new_tokens.append(int(new_token))
		wall_time.append(total_time)
		messages.append({
			"role": "assistant",
			"content": output
		})

	return turns, taus, new_tokens, wall_time, profiler


@torch.inference_mode()
def get_model_answers(
	base_model_path,
	spec_model_path,
	model_variant,
	model_id,
	questions,
	answer_file,
	max_new_token,
	num_choices,
):
	device = 'cuda'

	model = get_model(
		'paged_llama',
		model_variant,
		model_path=base_model_path,
		checkpoint_sharding=None,
		device_type=device,
		source='hf',
		distributed_strategy=None,
		group=None,
	)
	model.eval()

	#tokenizer = tokenizers.get_tokenizer(base_model_path)
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

	speculator = MLPSpeculatorPreTrainedModel.from_pretrained(spec_model_path, device_map=device).speculator

	kv_cache_manager = PagedKVCacheManager(
		model.config.nlayers,
		model.config.nheads,
		model.config.emb_dim,
		kv_heads=model.config.kvheads,
		tensor_parallel_size=1,
		dtype=torch.get_default_dtype(),
		device=device,
	)

	pad_token_id = tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("<|end_of_text|>")
	eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.eos_token_id is None or tokenizer.eos_token_id == pad_token_id else tokenizer.eos_token_id

	complete_method = complete_chat

	# warmup
	question0 = questions[0]
	for _ in tqdm(range(3), desc='Warming up'):
		torch.manual_seed(0)
		complete_method(
			question0,
			tokenizer,
			kv_cache_manager,
			model,
			speculator,
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
				kv_cache_manager,
				model,
				speculator,
				eos_token_id,
				pad_token_id,
				max_new_token,
			)

			#if not "base" in profiler:
			#	continue

			choices.append({
				"index": i,
				"turns": turns,
				"taus": taus,
				"new_tokens": new_tokens,
				"wall_time": wall_time,
				#"base_time": profiler.get("base", 0),
				#"ealayer_time": profiler.get("ea_layer", 0),
				#"head_time": profiler.get("head", 0),
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
	spec_model_path: Annotated[str, typer.Option('--spec-model-path', help='Path to the EAGLE model.')],
	data_path: Annotated[str, typer.Option('--data-path', help='Path to the data.')],
	postfix: Annotated[str, typer.Option('--postfix', help='Postfix on answer file name.')],
	max_new_token: Annotated[int, typer.Option('--max-new-token', help='Maximum number of new tokens to generate.')] = 512,
	model_variant: Annotated[str, typer.Option('--model-variant', help='Model variant.')] = 'llama3.8b'
	#by_instruction: Annotated[bool, typer.Option('--instruct', help='Use instruction data against chat.')] = False,
):
	model_id = base_model_path.split('/')[-1]

	data_name = os.path.basename(data_path)
	data_name = os.path.splitext(data_name)[0].replace('.local', '')

	os.makedirs(f'./evaluation/{model_id}', exist_ok=True)

	answer_file = f'./evaluation/{model_id}/{data_name}-mlpspec-{postfix}.jsonl'

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
			spec_model_path,
			model_variant,
			model_id,
			questions,
			answer_file,
			max_new_token,
			1,
		)
	)


if __name__ == '__main__':
	app()
