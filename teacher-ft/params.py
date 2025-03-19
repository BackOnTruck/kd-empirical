import torch, os, argparse
from transformers import ModernBertConfig, ModernBertForSequenceClassification, RobertaForSequenceClassification, RobertaConfig

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", choices=('0', '1', '2', '3'), required=True)
parser.add_argument("--task", choices=('defect', 'clone', 'except'), required=True)
parser.add_argument("--teacher", choices=('modernbert', 'unixcoder'), required=True)
parser.add_argument("--test_only", action='store_true')

args = parser.parse_args()
task, teacher, test_only = args.task, args.teacher, args.test_only

teacher_info = {
    'modernbert': (ModernBertForSequenceClassification, ModernBertConfig, 'model/modernbert-base'),
    'unixcoder': (RobertaForSequenceClassification, RobertaConfig, 'model/unixcoder-base'),
}[teacher]
t_cls, tcfg_cls, ORIGINAL_MODEL = teacher_info

DATASET_ROOT = {
    'defect': 'data/Defect-detection/dataset',
    'clone': 'data/Clone-detection-BigCloneBench/dataset',
    'except': 'data/Exception-classification'
}[task]

TRAIN_DATASET = f"{DATASET_ROOT}/train.jsonl"
VALID_DATASET = f"{DATASET_ROOT}/valid.jsonl"
TEST_DATASET = f"{DATASET_ROOT}/test.jsonl"

BLOCK_SIZE = 512
BATCH_SIZE = 32
EPOCH = 5

LEARNING_RATE = 2e-5
ADAM_EPSILON = 1e-8

TUNED_MODEL = f"model/{teacher}-tuned-{task}-{LEARNING_RATE}"
print(f"{args}\n# of epoch = {EPOCH}, Batch size = {BATCH_SIZE}, Block size = {BLOCK_SIZE}, Learning rate = {LEARNING_RATE}")

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # only supports a single GPU
DEVICE = torch.device("cuda")
ALT_DEVICE = torch.device("cuda:1")
