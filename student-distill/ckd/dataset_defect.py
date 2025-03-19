from torch import tensor, Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
from typing import *
from tqdm import tqdm

from params import *

class TextDataset(Dataset):
    def __init__(self, dataset_file: str, test_only: bool):
        tk = AutoTokenizer.from_pretrained(NEW_TK)
        t_tk = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
        tk.do_lowercase = t_tk.do_lowercase = True
        self.label, self.data, self.t_data = [], [], []

        with open(dataset_file) as f:
            for line in tqdm(f, desc=">>> Loading data"):
                js = json.loads(line)
                func = ' '.join(js['func'].split())
                res = tk(func, padding='max_length', truncation=True,
                    return_attention_mask=True, return_token_type_ids=True, return_tensors='pt', max_length=512)
                t_res = t_tk(func, padding='max_length', truncation=True,
                    return_attention_mask=True, return_token_type_ids=True, return_tensors='pt') \
                    if not test_only else {'input_ids': [[]], 'attention_mask': [[]], 'token_type_ids': [[]]}

                if teacher == 'unixcoder' and not test_only:
                    prep_unixcoder = lambda a,b: torch.cat([a[:1], b, a[1:]])
                    # [CLS] <encoder-only> [SEP] ...contents... (<encoder-only>: 6, [SEP]: 2)
                    t_res['input_ids'] = prep_unixcoder(t_res['input_ids'][0], tensor([6, 2])).unsqueeze(0)
                    t_res['attention_mask'] = prep_unixcoder(t_res['attention_mask'][0], tensor([1, 1])).unsqueeze(0)
                    t_res['token_type_ids'] = prep_unixcoder(t_res['token_type_ids'][0], tensor([0, 0])).unsqueeze(0)

                self.label += [js['target']]
                self.data += [[res['input_ids'][0], res['attention_mask'][0], res['token_type_ids'][0]]]
                self.t_data += [[t_res['input_ids'][0], t_res['attention_mask'][0], t_res['token_type_ids'][0]]]

        print(f">>> {len(self.data)} samples loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        # format: label, inputs, masks, token_types, t_inputs, t_masks, t_token_types
        return [tensor(self.label[index])] + self.data[index] + self.t_data[index]
