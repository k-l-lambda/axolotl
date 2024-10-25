
import json
from transformers import AutoTokenizer
import numpy as np
import argparse



def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('--tokenizer', type=str)
	parser.add_argument('--jsonl-base', type=str)
	parser.add_argument('--postfix', type=str, default='eagle:baseline')
	args = parser.parse_args()


	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

	postfix1, postfix_base = args.postfix.split(':')
	jsonl_file = f'{args.jsonl_base}{postfix1}.jsonl'
	jsonl_file_base = f'{args.jsonl_base}{postfix_base}.jsonl'


	data = []
	with open(jsonl_file, 'r', encoding='utf-8') as file:
		for line in file:
			json_obj = json.loads(line)
			data.append(json_obj)



	speeds = []
	taus = []
	durations = dict(
		total=0,
		base=0,
		ealayer=0,
		head=0,
		other=0,
	)

	for datapoint in data:
		#qid=datapoint['question_id']
		if len(datapoint['choices']) <= 0:
			continue
		answer = datapoint['choices'][0]['turns']
		tokens = sum(datapoint['choices'][0]['new_tokens'])
		times = sum(datapoint['choices'][0]['wall_time'])
		speeds.append(tokens/times)
		taus += datapoint['choices'][0]['taus']

		durations['total'] += times
		if 'base_time' in datapoint['choices'][0]:
			durations['base'] += datapoint['choices'][0]['base_time']
			durations['ealayer'] += datapoint['choices'][0]['ealayer_time']
			durations['head'] += datapoint['choices'][0]['head_time']
			durations['other'] += times - (datapoint['choices'][0]['head_time'] + datapoint['choices'][0]['ealayer_time'] + datapoint['choices'][0]['base_time'])


	data = []
	with open(jsonl_file_base, 'r', encoding='utf-8') as file:
		for line in file:
			json_obj = json.loads(line)
			data.append(json_obj)


	total_time = 0
	total_token = 0
	speeds0 = []
	for datapoint in data:
		#qid=datapoint['question_id']
		answer = datapoint['choices'][0]['turns']
		tokens = 0
		for i in answer:
			tokens += (len(tokenizer(i).input_ids) - 1)
		times = sum(datapoint['choices'][0]['wall_time'])
		if times <= 0:
			continue
		speeds0.append(tokens / times)
		total_time += times
		total_token += tokens



	tau = np.array(taus).mean()
	print('speed', np.array(speeds).mean())
	# print('speed0',np.array(speeds0).mean())
	print('ratio', np.array(speeds).mean() / np.array(speeds0).mean())
	print('tau', tau)
	print('durations', durations)

	mean_durations = {k: v * 1000 / len(taus) for k, v in durations.items()}
	print('mean_durations', mean_durations)
	print('draft latency per token:', (mean_durations['ealayer'] + mean_durations['head'] + mean_durations['other']) / tau)


if __name__ == '__main__':
	main()
