
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM
import wandb



def load_teacher_model (self, model_id: str, **_):
	self.teacher_model = AutoModelForCausalLM.from_pretrained(model_id)
	self.teacher_model.to(self.dtype).to(self.device)


def teacher_replace_compute_loss(distill_coef):
	original_compute_loss = transformers.trainer.Trainer.compute_loss

	def compute_loss(self, model, inputs, return_outputs=False):
		loss_fct, output = original_compute_loss(self, model, inputs, return_outputs=True)

		teacher_model = model.teacher_model if hasattr(model, 'teacher_model') else model.module.teacher_model
		teacher_logits = teacher_model(**inputs).logits
		loss_distill = F.kl_div(F.log_softmax(output.logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='sum')

		loss = loss_fct + loss_distill * distill_coef

		log = {
			'loss_fct': loss_fct,
			'loss_distill': loss_distill,
		}
		if self.state.is_world_process_zero:
			wandb.log({
				**log,
				"train/global_step": self.state.global_step,
			})

		return (loss, output) if return_outputs else loss

	transformers.trainer.Trainer.compute_loss = compute_loss
