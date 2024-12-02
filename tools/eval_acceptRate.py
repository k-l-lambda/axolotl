
import typer
from typing_extensions import Annotated
from tqdm import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle



app = typer.Typer(pretty_exceptions_show_locals=False)


def test_question (tokenizer, model, question):
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

			if msg['role'] == 'user':
				n_prefix = tokens.shape[-1]
			else:
				logits = model(tokens.cuda()).logits[0, n_prefix:].cpu()
				n_total += logits.shape[0]

				acception = logits.argmax(dim=-1) == tokens[0, n_prefix:]
				n_accepted += acception.sum().item()

	return n_total, n_accepted



@app.command()
def run_eval (
	model_path: Annotated[str, typer.Option('--model-path', help='Path to the base model.')],
	data_path: Annotated[str, typer.Option('--data-path', help='Path to the data.')],
):
	questions = datasets.Dataset.from_json(data_path)

	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cuda')

	n_tokens, n_accepted_tokens = 0, 0
	with tqdm(questions) as progress_bar:
		for question in progress_bar:
			t, a = test_question(tokenizer, model, question)
			n_tokens += t
			n_accepted_tokens += a

			progress_bar.set_description(f'{n_accepted_tokens}/{n_tokens} = {n_accepted_tokens/n_tokens:.4f}')

	accept_rate = n_accepted_tokens / n_tokens
	print(f'{accept_rate=}')
	print(f'{n_tokens=}, {n_accepted_tokens=}')


if __name__ == '__main__':
	app()
