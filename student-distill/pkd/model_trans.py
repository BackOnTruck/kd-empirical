from torch import nn, Tensor
from transformers import RobertaForSequenceClassification as RSC, RobertaConfig as RC
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaEmbeddings as REmbed, RobertaClassificationHead as RCH, RobertaLayer as RL
from typing import *
import torch.nn.functional as F

from params import *

class ModelOutput:
    def __init__(self, logits: Tensor, loss: Optional[Tensor]):
        self.logits = logits
        if loss is not None:
            self.loss = loss

class MultiLayered(nn.Module):
    def __init__(self, mods: List[RL]):
        super().__init__()
        self.mods = nn.ModuleList(mods)

    def forward(self, *args):
        res, others = args[0], args[1:]
        for trans in self.mods: res = trans(res, *others)[0]
        return [res]

    def load_state_dict(self, *args):
        return self.mods.load_state_dict(*args)

class Student(RobertaPreTrainedModel):
    def __init__(self, config: RC, model: RSC):
        super().__init__(config)
        self.config, large_cfg = config, RC.from_pretrained(TUNED_MODEL)

        t_hidden, s_hidden = large_cfg.hidden_size, self.config.hidden_size
        t_layer, s_layer = large_cfg.num_hidden_layers, self.config.num_hidden_layers
        assert t_layer % s_layer == 0, f'Invalid number of layers: {t_layer} and {s_layer}'
        size = t_layer // s_layer

        self.t_mods, self.mods = [], nn.ModuleList([])

        self.add_module(REmbed(large_cfg), model.roberta.embeddings, REmbed(config))
        for i in range(s_layer):
            self.add_module(MultiLayered([RL(large_cfg) for _ in range(size)]),
                model.roberta.encoder.layer[i * size:(i + 1) * size],
                RL(config))

        self.add_module(RCH(large_cfg), model.classifier, RCH(config))
        self.interm = nn.ModuleList([nn.Linear(t_hidden, s_hidden) for _ in range(s_layer)])
        self.post_init()

    def to(self, device):
        for t_mod in self.t_mods: t_mod.to(device)
        return super().to(device)

    def add_module(self, large_init: nn.Module, large: nn.Module, small: nn.Module):
        self.mods.append(small)
        self.t_mods.append(large_init)
        self.t_mods[-1].load_state_dict(large.state_dict())

        for param in self.t_mods[-1].parameters():
            param.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor,
            t_inputs: Optional[Tensor], t_masks: Optional[Tensor], t_ids: Optional[Tensor],
            labels: Optional[Tensor]):
        ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
        mode = labels is not None

        attn_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())
        t_attn_mask = self.get_extended_attention_mask(t_masks, t_inputs.size()) if mode else None

        res = self.mods[0](input_ids, token_type_ids)
        t_res = self.t_mods[0](t_inputs, t_ids) if mode else None
        feat, t_feat = [], [] # [n_layers, batch_size, hidden_size] before transposing; use features at [CLS] token

        for (mod, t_mod, interm) in zip(self.mods[1:-1], self.t_mods[1:-1], self.interm):
            res = mod(res, attn_mask)[0]
            t_res = t_mod(t_res, t_attn_mask)[0] if mode else None
            feat += [res[:, 0, :]]
            t_feat += [interm(t_res[:, 0, :]) if mode else None]

        res = self.mods[-1](res)
        t_res = self.t_mods[-1](t_res) if mode else None
        feat = torch.stack(feat[:-1]).transpose(0, 1)
        t_feat = torch.stack(t_feat[:-1]).transpose(0, 1) if mode else None

        norm = lambda x: F.normalize(x, p=2, dim=2)
        if mode: return res, ce(res / TEMP, F.softmax(t_res / TEMP, dim=-1)) * TEMP ** 2 * ALPHA + \
            ce(res, labels) * (1 - ALPHA) + mse(norm(feat), norm(t_feat)) * BETA
        return res, None

class SavedModel(RobertaPreTrainedModel):
    def __init__(self, config: RC):
        super().__init__(config)

        self.embed = REmbed(config)
        self.layers = nn.ModuleList([RL(config) for _ in range(config.num_hidden_layers)])
        self.classifier = RCH(config)

        self.post_init()

    @staticmethod
    def load_student(config: RC, model: Student):
        mod = SavedModel(config)

        mod.embed.load_state_dict(model.mods[0].state_dict())
        for i, layer in enumerate(mod.layers, start=1):
            layer.load_state_dict(model.mods[i].state_dict())

        mod.classifier.load_state_dict(model.mods[-1].state_dict())
        return mod

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, *args):
        res = self.embed(input_ids=input_ids, token_type_ids=token_type_ids)
        ext_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())

        for layer in self.layers:
            res = layer(res, ext_mask)[0]

        return self.classifier(res), None

class Model(nn.Module):
    def __init__(self, from_scratch: bool):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

        if from_scratch:
            print(">>> Initializing student model...")
            large = RSC.from_pretrained(TUNED_MODEL)
            self.cfg = RC.from_json_file(DISTILL_CONFIG)
            if task == 'except':
                self.cfg.num_labels = 20

            self.model = Student(self.cfg, large)

        else:
            self.model = SavedModel.from_pretrained(DISTILLED_MODEL)

    def to(self, device):
        return self.model.to(device)

    def save_pretrained(self, dir: str):
        SavedModel.load_student(self.cfg, self.model).save_pretrained(dir)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor,
            t_inputs: Optional[Tensor]=None, t_masks: Optional[Tensor]=None, t_ids: Optional[Tensor]=None,
            labels: Optional[Tensor]=None):
        return ModelOutput(*self.model(input_ids, attention_mask, token_type_ids, t_inputs, t_masks, t_ids, labels))

def param_count(mod: nn.Module):
    return sum([p.numel() for p in mod.parameters()])

if __name__ == '__main__':
    s_mod = SavedModel(RC.from_json_file(DISTILL_CONFIG))
    t_mod = RSC.from_pretrained(TUNED_MODEL)

    s_num, t_num, s_embed, t_embed, s_tr, t_tr, s_head, t_head = map(param_count,
        (s_mod, t_mod, s_mod.embed, t_mod.roberta.embeddings, s_mod.layers, t_mod.roberta.encoder, s_mod.classifier, t_mod.classifier))
    print(f'Student: {s_num:,} parameters (embed: {s_embed:,}, transformer: {s_tr:,}, classifier: {s_head:,})')
    print(f'Teacher: {t_num:,} parameters (embed: {t_embed:,}, transformer: {t_tr:,}, classifier: {t_head:,})')
    print(f'Size reduced by {(1 - s_num / t_num) * 100:.2f}%')
