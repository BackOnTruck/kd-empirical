from torch import nn, Tensor
from transformers import RobertaForSequenceClassification as RSC, RobertaConfig as RC
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaEmbeddings as REmbed, RobertaClassificationHead as RCH, RobertaLayer as RL
from typing import *
import torch.nn.functional as F

from params import *

RNN_HIDDEN, RNN_LAYERS = (176, 2) if args.tokenizer == '2000' else (320, 3) if args.tokenizer == '5000' else (448, 6)

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

def p_corr(x: Tensor, y: Tensor):
    sim = nn.CosineSimilarity()
    return sim(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1)).mean()

class Student(RobertaPreTrainedModel):
    def __init__(self, config: RC, model: RSC):
        config.hidden_size, config.num_hidden_layers = RNN_HIDDEN, RNN_LAYERS
        super().__init__(config)
        self.config, large_cfg = config, RC.from_pretrained(TUNED_MODEL)
        self.t_mods, self.mods = [], nn.ModuleList([])

        self.add_module(REmbed(large_cfg), model.roberta.embeddings, REmbed(config))
        self.add_module(MultiLayered([RL(large_cfg) for _ in range(large_cfg.num_hidden_layers)]),
            model.roberta.encoder.layer,
            nn.GRU(input_size=RNN_HIDDEN, hidden_size=RNN_HIDDEN, num_layers=RNN_LAYERS, bidirectional=True, batch_first=True))

        config.hidden_size *= 2
        self.add_module(RCH(large_cfg), model.classifier, RCH(config))
        config.hidden_size //= 2

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
        loss = nn.CrossEntropyLoss()
        mode = labels is not None

        t_attn_mask = self.get_extended_attention_mask(t_masks, t_inputs.size()) if mode and not no_distill else None

        res = self.mods[0](input_ids, token_type_ids)
        t_res = self.t_mods[0](t_inputs, t_ids) if mode and not no_distill else None
        for (mod, t_mod) in zip(self.mods[1:-1], self.t_mods[1:-1]):
            res = mod(res)[0]
            t_res = t_mod(t_res, t_attn_mask)[0] if mode and not no_distill else None

        res = self.mods[-1](torch.cat((res[:, -1, :RNN_HIDDEN], res[:, 0, RNN_HIDDEN:]), dim=1).unsqueeze(1))
        t_res = self.t_mods[-1](t_res) if mode and not no_distill else None

        if mode:
            if no_distill:
                return res, loss(res, labels)

            p_s, p_t = F.softmax(res / TEMP, dim=-1), F.softmax(t_res / TEMP, dim=-1)
            return res, BETA * (1 - p_corr(p_s, p_t)) + GAMMA * (1 - p_corr(p_s.T, p_t.T))

        return res, None

class SavedModel(RobertaPreTrainedModel):
    def __init__(self, config: RC):
        super().__init__(config)

        self.embed = REmbed(config)
        self.layers = nn.GRU(input_size=RNN_HIDDEN, hidden_size=RNN_HIDDEN, num_layers=RNN_LAYERS, bidirectional=True, batch_first=True)

        config.hidden_size *= 2
        self.classifier = RCH(config)
        config.hidden_size //= 2

        self.post_init()

    @staticmethod
    def load_student(config: RC, model: Student):
        mod = SavedModel(config)

        mod.embed.load_state_dict(model.mods[0].state_dict())
        mod.layers.load_state_dict(model.mods[1].state_dict())
        mod.classifier.load_state_dict(model.mods[-1].state_dict())

        return mod

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, *args):
        res = self.layers(self.embed(input_ids=input_ids, token_type_ids=token_type_ids))[0]

        return self.classifier(torch.cat((res[:, -1, :RNN_HIDDEN], res[:, 0, RNN_HIDDEN:]), dim=1).unsqueeze(1)), None

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
    config = RC.from_json_file(DISTILL_CONFIG)
    config.hidden_size, config.num_hidden_layers = RNN_HIDDEN, RNN_LAYERS
    s_mod = SavedModel(config)
    t_mod = RSC.from_pretrained(TUNED_MODEL)

    s_num, t_num, s_embed, t_embed, s_tr, t_tr, s_head, t_head = map(param_count,
        (s_mod, t_mod, s_mod.embed, t_mod.roberta.embeddings, s_mod.layers, t_mod.roberta.encoder, s_mod.classifier, t_mod.classifier))
    print(f'Student: {s_num:,} parameters (embed: {s_embed:,}, transformer: {s_tr:,}, classifier: {s_head:,})')
    print(f'Teacher: {t_num:,} parameters (embed: {t_embed:,}, transformer: {t_tr:,}, classifier: {t_head:,})')
    print(f'Size reduced by {(1 - s_num / t_num) * 100:.2f}%')
