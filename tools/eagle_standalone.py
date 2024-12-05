
import torch
from dataclasses import dataclass

from axolotl.models.eagle import EaModel



@dataclass
class LMOutput:
	logits: torch.Tensor


class EagleStandalone (torch.nn.Module):
	def __init__(self, ea_model):
		super().__init__()

		self.ea_layer = ea_model.ea_layer
		self.base_model = ea_model.base_model
		self.tokenizer = ea_model.tokenizer

		self.base_model.model.embed_tokens.to('cuda')


	@classmethod
	def from_pretrained (
			cls,
			Type="LLaMA",
			base_model_path=None,
			ea_model_path=None,
	):
		ea_model = EaModel.from_pretrained(
			Type=Type,
			base_model_path=base_model_path,
			ea_model_path=ea_model_path,
			torch_dtype=torch.bfloat16,
			low_cpu_mem_usage=True,
			device_map="cuda",
		)
		ea_model.eval()
		return cls(ea_model)


	@torch.inference_mode()
	def forward (self, input_ids, attention_mask=None, past_key_values=None):
		last_hidden_state = self.base_model.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values).last_hidden_state
		last_hidden_state = last_hidden_state.roll(shifts=1, dims=1)

		hidden_state = self.ea_layer(last_hidden_state, input_ids=input_ids)

		logits = self.base_model.lm_head(hidden_state)

		return LMOutput(logits=logits)
