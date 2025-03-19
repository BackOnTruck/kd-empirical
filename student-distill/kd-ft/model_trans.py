from torch import nn, Tensor
from transformers import RobertaForSequenceClassification as RSC, RobertaConfig as RC, ModernBertForSequenceClassification as MBSC
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaEmbeddings as REmbed, RobertaClassificationHead as RCH, RobertaLayer as RL
from typing import *
import torch.nn.functional as F

from params import *

class ModelOutput:
    def __init__(self, logits: Tensor, loss: Optional[Tensor]):
        self.logits = logits
        if loss is not None:
            self.loss = loss

class Student(RobertaPreTrainedModel):
    def __init__(self, config: RC, model: RSC):
        super().__init__(config)
        self.config = config
        self.model = RSC(config)

        self.large = [model]
        for param in model.parameters():
            param.requires_grad = False

        self.post_init()

    def to(self, device):
        self.large[0].to(device)
        return super().to(device)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor,
            t_inputs: Optional[Tensor], t_masks: Optional[Tensor], t_ids: Optional[Tensor],
            labels: Optional[Tensor]):
        loss = nn.CrossEntropyLoss()
        mode = labels is not None

        res = self.model(input_ids, attention_mask, token_type_ids)[0]
        t_res = self.large[0](t_inputs, t_masks, t_ids)[0] if mode and not no_distill else None

        if mode: return res, loss(res / TEMP, F.softmax(t_res / TEMP, dim=-1)) if not no_distill else loss(res, labels)
        return res, None

class Model(nn.Module):
    def __init__(self, from_scratch: bool):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

        if from_scratch:
            print(">>> Initializing student model...")
            t_cls = RSC if teacher == 'unixcoder' else MBSC
            large = t_cls.from_pretrained(TUNED_MODEL)

            self.cfg = RC.from_json_file(DISTILL_CONFIG)
            if task == 'except':
                self.cfg.num_labels = 20

            self.model = Student(self.cfg, large)

        else:
            self.model = RSC.from_pretrained(DISTILLED_MODEL)

    def to(self, device):
        return self.model.to(device)

    def save_pretrained(self, dir: str):
        self.model.model.save_pretrained(dir)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor,
            t_inputs: Optional[Tensor]=None, t_masks: Optional[Tensor]=None, t_ids: Optional[Tensor]=None,
            labels: Optional[Tensor]=None):

        if isinstance(self.model, Student):
            return ModelOutput(*self.model(input_ids, attention_mask, token_type_ids, t_inputs, t_masks, t_ids, labels))

        return ModelOutput(self.model(input_ids, attention_mask, token_type_ids)[0], None)

def param_count(mod: nn.Module):
    return sum([p.numel() for p in mod.parameters()])

if __name__ == '__main__':
    s_mod = RSC(RC.from_json_file(DISTILL_CONFIG))
    t_mod = RSC.from_pretrained(TUNED_MODEL)

    s_num, t_num, s_embed, t_embed, s_tr, t_tr, s_head, t_head = map(param_count,
        (s_mod, t_mod, s_mod.embed, t_mod.roberta.embeddings, s_mod.layers, t_mod.roberta.encoder, s_mod.classifier, t_mod.classifier))
    print(f'Student: {s_num:,} parameters (embed: {s_embed:,}, transformer: {s_tr:,}, classifier: {s_head:,})')
    print(f'Teacher: {t_num:,} parameters (embed: {t_embed:,}, transformer: {t_tr:,}, classifier: {t_head:,})')
    print(f'Size reduced by {(1 - s_num / t_num) * 100:.2f}%')
