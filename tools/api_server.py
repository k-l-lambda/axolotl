import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
from flask import Flask, request, jsonify
import argparse

from .throttle_lock import NSlotLock



# Parse command-line arguments
parser = argparse.ArgumentParser(description='transformers API Server')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind the server')
parser.add_argument('--port', type=int, default=7865, help='Port number to run the server on')
parser.add_argument('--model-path', type=str, required=True, help='Path to the base model')
args = parser.parse_args()


app = Flask(__name__)

device_table = {}


def worker (device_index, qin, qout):
	device = f'cuda:{device_index}'

	tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

	model = AutoModelForCausalLM.from_pretrained(
		args.model_path,
		torch_dtype=torch.float16,
		device_map=device,
	)
	model.eval()

	qout.put('ready')

	eos_token_id = tokenizer.eos_token_id
	if eos_token_id == 128001: # set EOS token for llama3
		tokenizer.pad_token_id = eos_token_id
		eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

	if tokenizer.pad_token_id is None: # set PAD token for llama3.1
		tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

	while True:
		input = qin.get()
		if input is None:
			break

		# Create a conversation template based on the model type
		conv = get_conversation_template(args.model_path)

		# Add messages to the conversation template
		for message in input['messages']:
			role = message.get('role', '')
			content = message.get('content', None)
			conv.append_message(role, content)
		conv.append_message(conv.roles[1], '')

		# Get the prompt from the conversation template
		prompt = conv.get_prompt()

		input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(device)

		temperature = input['temperature']

		try:
			with torch.inference_mode():
				output_ids = model.generate(
					input_ids=input_ids,
					do_sample=temperature > 0,
					temperature=temperature if temperature > 0 else None,
					top_p=input['top_p'],
					top_k=input['top_k'],
					max_new_tokens=input['max_new_tokens'],
					eos_token_id=eos_token_id,
					pad_token_id=tokenizer.pad_token_id,
				)

			generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

			qout.put(generated_text)
		except Exception as e:
			print('error in eagenerate:', e)
			qout.put(None)


@app.route('/generate', methods=['POST'])
async def generate ():
	async with NSlotLock(device_table, num_gpus) as lock:
		print('got device:', lock.key)

		data = request.get_json()
		messages = data.get('messages', [])
		max_new_tokens = data.get('max_new_tokens', 512)
		temperature = data.get('temperature', 0.0)
		top_p = data.get('top_p', 0.0)
		top_k = data.get('top_k', 0)

		input = dict(
			messages=messages,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
			top_p=top_p,
			top_k=top_k,
		)

		qin, qout = device_queues[lock.key]
		qin.put(input)
		output = qout.get()

		if output is None:
			return jsonify({'error': 'failed to generate'}), 500

		return jsonify({'choices': [
			{'message': {
				'role': 'assistant',
				'content': output,
			}},
		]})


if __name__ == '__main__':
	num_gpus = torch.cuda.device_count()
	request_i = 0
	device_queues = {}
	processes = []

	mp.set_start_method('spawn', force=True)

	for i in range(num_gpus):
		qin, qout = mp.Queue(), mp.Queue()
		device_queues[i] = (qin, qout)
		p = mp.Process(target=worker, args=(i, qin, qout))
		p.start()
		processes.append(p)

	for i, (qin, qout) in device_queues.items():
		init_state = qout.get()
		print(f'device[{i}]: {init_state}')

	app.run(host=args.host, port=args.port, debug=False, threaded=True)

	# Cleanup
	for i in range(num_gpus):
		device_queues[i][0].put(None)
	for p in processes:
		p.join()
