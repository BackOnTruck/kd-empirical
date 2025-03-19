from torch import tensor, Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
from typing import *
from tqdm import tqdm

from params import *

class TextDataset(Dataset):
    def __init__(self, dataset_file: str):
        tk = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
        tk.do_lowercase = True
        self.label, self.data = [], []

        with open(dataset_file) as f:
            for line in tqdm(f, desc=">>> Loading data"):
                js = json.loads(line)
                func = ' '.join(js['func'].split())
                res = tk(func, padding='max_length', truncation=True,
                    return_attention_mask=True, return_token_type_ids=True, return_tensors='pt')

                if teacher == 'unixcoder':
                    prep_unixcoder = lambda a,b: torch.cat([a[:1], b, a[1:]])
                    # [CLS] <encoder-only> [SEP] ...contents... (<encoder-only>: 6, [SEP]: 2)
                    res['input_ids'] = prep_unixcoder(res['input_ids'][0], tensor([6, 2])).unsqueeze(0)
                    res['attention_mask'] = prep_unixcoder(res['attention_mask'][0], tensor([1, 1])).unsqueeze(0)
                    res['token_type_ids'] = prep_unixcoder(res['token_type_ids'][0], tensor([0, 0])).unsqueeze(0)

                self.label += [js['target']]
                self.data += [[res['input_ids'][0], res['attention_mask'][0], res['token_type_ids'][0]]]

        print(f">>> {len(self.data)} samples loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        # format: label, inputs, masks, token_types
        return [tensor(self.label[index])] + self.data[index]
