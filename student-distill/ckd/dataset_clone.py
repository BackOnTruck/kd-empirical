from torch import tensor, Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json, os
from typing import *
from tqdm import tqdm
from random import random

from params import *

CACHE_FILE = f'{DATASET_ROOT}/cache-{args.tokenizer}.pt' # only used for test set where self.t_data is unused
class TextDataset(Dataset):
    def __init__(self, dataset_file: str, test_only: bool):
        tk = AutoTokenizer.from_pretrained(NEW_TK)
        t_tk = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
        tk.do_lowercase = t_tk.do_lowercase = True
        self.label, self.data, self.t_data = [], [], []

        use_all = dataset_file == TEST_DATASET
        if use_all and os.path.isfile(CACHE_FILE):
            print(">>> Loading cached data...")
            self.label, self.data, self.t_data = torch.load(CACHE_FILE)
            print(f">>> {len(self.data)} samples loaded")
            return

        with open(dataset_file) as f:
            for line in tqdm(f, desc=">>> Loading data"):
                # following CodeXGLUE, we use 10% train/valid data and 100% test data for Clone Detection
                if not use_all and random() > 0.1:
                    continue

                js = json.loads(line)
                func1, func2 = ' '.join(js['func1'].split()), ' '.join(js['func2'].split())
                res = tk(func1, func2, padding='max_length', truncation=True,
                    return_attention_mask=True, return_token_type_ids=True, return_tensors='pt', max_length=512)
                t_res = t_tk(func1, func2, padding='max_length', truncation=True,
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
        if use_all:
            print(">>> Saving cached data...")
            with open(CACHE_FILE, 'wb') as f:
                torch.save([self.label, self.data, self.t_data], f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        # format: label, inputs, masks, token_types, t_inputs, t_masks, t_token_types
        return [tensor(self.label[index])] + self.data[index] + self.t_data[index]

if __name__ == '__main__':
    print(f">>> Caching test set for {NEW_TK}...")
    _ = TextDataset(TEST_DATASET, True)
