from params import *

if task in ("defect", 'except'):
    from dataset_defect import TextDataset
elif task == "clone":
    from dataset_clone import TextDataset
