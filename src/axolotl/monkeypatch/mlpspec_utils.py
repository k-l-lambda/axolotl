
from typing import List, Optional
import torch
from torch.nn import CrossEntropyLoss
import types
import transformers
import wandb

from axolotl.models.fms_extras.models.hf.modeling_mlp_speculator import MLPSpeculatorPreTrainedModel, MLPSpeculatorConfig



def add_mlpspec_speculator (self,
	emb_dim=4096,
	inner_dim=3072,
	n_candidates=5,
	n_predict=4,
	vocab_size=128256,
	top_k_tokens_per_head=[4, 3, 2, 2],
	**_,
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

		base_out = self.model(
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

		outputs = self.mlp_model(base_out.last_hidden_state[:, :-n_predict], input_ids)

		return outputs

	self.forward = types.MethodType(forward, self)



def mlpspec_replace_compute_loss(
	decay_coefficient=0.8,
):
	def compute_loss(self, model, inputs, return_outputs=False):
		"""
		Compute the training loss for the model.

		Args:
			model (torch.nn.Module): The model for which to compute the loss.
			inputs (dict): The input data, including input IDs, attention mask, and labels.
			return_outputs (bool): Whether to return model outputs along with the loss.

		Returns:
			Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
		"""
		logits = model(
			**inputs,
			mlpspec_return=True,
		)
		labels = inputs["labels"]
		loss = 0
		loss_fct = CrossEntropyLoss()
		log = {}

		n_logits = logits.shape[2]
		n_predict = logits.shape[0]
		for i in range(n_predict):
			medusa_logits = logits[i, :, : -(1 + i)].contiguous()
			medusa_labels = labels[..., 1 + i :n_logits].contiguous()
			medusa_logits = medusa_logits.view(-1, logits.shape[-1])
			medusa_labels = medusa_labels.view(-1)
			medusa_labels = medusa_labels.to(medusa_logits.device)

			loss_i = loss_fct(medusa_logits, medusa_labels)
			if i == 0:
				loss += loss_i
			else:
				loss += loss_i * decay_coefficient ** i

			log[f"mlpspec{i}_loss"] = loss_i.item()

		# Add prefix to the log
		if model.training:
			prefix = "train"
		else:
			prefix = "eval"

		log = {f"{prefix}/{k}": v for k, v in log.items()}
		if self.state.is_world_process_zero:
			# Hardcoded for now
			wandb.log({
				**log,
				"train/global_step": self.state.global_step,
			})
		return (loss, logits) if return_outputs else loss

	transformers.trainer.Trainer.compute_loss = compute_loss
