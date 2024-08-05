
import typer
from typing_extensions import Annotated
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm



app = typer.Typer()


prompt = """Classify the given chat message into one of some topic categories.
* "I’m having trouble with the wireless headphones I bought last week. They won’t connect to my phone."→Customer Support.
* "I need help planning a trip to New York next month."→Personal Assistance.
* "I'm struggling with logarithms in my math class."→Education and Learning.
* "I’m in the mood for a movie tonight. Got any recommendations?"→Entertainment.
* "I’ve also heard about the significance of Pearl Harbor. Can you explain that?"→General Knowledge.
* """


@app.command()
def main (source_dataset,
		  model_name: Annotated[str, typer.Option("--model")],
):
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
	model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.float16)

	model.generation_config.pad_token_id = tokenizer.pad_token_id

	output_file = open('./outputs/topic_annotator.tsv', 'w')
	output_file.write('topic	query\n')

	source = json.load(open(source_dataset, 'r'))
	for ex in tqdm(source):
		conversations = ex['conversations']
		query = conversations[0]['value']
		query = query[:256]
		query_escaped = query.replace('"', '\\"')

		prompt_text = f'{prompt}"{query_escaped}"→'
		#print('prompt_text:', prompt_text)
		input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
		in_len = input_ids.shape[-1]

		with torch.no_grad():
			output = model.generate(input_ids, max_new_tokens=5, num_return_sequences=1, do_sample=False)
		output_ids = output[:, in_len:]
		result = tokenizer.decode(output_ids[0].tolist())
		result = result.split('\n')[0]
		result = result.split('.')[0]
		result = result.split('"')[0]
		#print(f'{result=}, {query=}')
		query_to_log = json.dumps(query.replace('"', '')).replace('\t', ' ')
		output_file.write(f'{result}	{query_to_log}\n')


if __name__ == "__main__":
	app()
