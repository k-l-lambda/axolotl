
from typing import List, Optional
import torch
import types

from axolotl.models.fms_extras.models.hf.modeling_mlp_speculator import MLPSpeculatorPreTrainedModel, MLPSpeculatorConfig



def add_mlpspec_speculator (self,
	emb_dim=4096,
	inner_dim=3072,
	n_candidates=5,
	n_predict=4,
	vocab_size=128256,
	top_k_tokens_per_head=[4, 3, 2, 2],
):
	config = MLPSpeculatorConfig(
		vocab_size=vocab_size,
		emb_dim=emb_dim,
		inner_dim=inner_dim,
		n_predict=n_predict,
		top_k_tokens_per_head=top_k_tokens_per_head,
		n_candidates=n_candidates,
	)

	self.mlp_model = MLPSpeculatorPreTrainedModel(config)

	self.mlp_model.to(self.dtype).to(self.device)

	# inner_dim is different from hidden_state_size of base model, so copying is not possible
	#for head in self.mlp_model.speculator.head:
	#	head.weight.data[:] = self.lm_head.weight.data[:]

	self.old_forward = self.forward

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		mlpspec_return: bool = False,
	):
		if not mlpspec_return:
			return self.old_forward(
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_values=past_key_values,
				inputs_embeds=inputs_embeds,
				use_cache=use_cache,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)

		hidden_states = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		outputs = self.mlp_model(hidden_states, input_ids)

		return outputs

	self.forward = types.MethodType(forward, self)
