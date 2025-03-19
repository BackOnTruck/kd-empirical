import torch, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", choices=('0', '1', '2', '3'), required=True)
parser.add_argument("--task", choices=('defect', 'clone', 'except'), required=True)
parser.add_argument("--teacher", choices=('modernbert', 'unixcoder'))
parser.add_argument('--student', choices=('roberta', 'gru'), required=True)
parser.add_argument('--tokenizer', choices=('1000', '2000', '5000', '20000', '50000'), required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--temp', type=float, default=10.0)
parser.add_argument("--test_only", action='store_true')
parser.add_argument("--no_distill", action='store_true')

args = parser.parse_args()
task, teacher, test_only, no_distill = args.task, args.teacher, args.test_only, args.no_distill

if not teacher:
    assert no_distill, 'Teacher model not provided for distillation'
    teacher = 'unixcoder' # avoid KeyError

ORIGINAL_MODEL = {
    'modernbert': 'model/modernbert-base',
    'unixcoder': 'model/unixcoder-base'
}[teacher] # pretrained large model

TUNED_MODEL = f"model/{teacher}-tuned-{task}"
DISTILLED_MODEL = f"model/distilled-{task}-{teacher}-{args.student}-{args.tokenizer}-{args.lr}-{args.temp}" \
    if not no_distill else f'model/raw-{task}-{args.student}-{args.tokenizer}-{args.lr}'

TK_SUFFIX = {
    'defect': '',
    'clone': '',
    'except': '-exc'
}[task]
NEW_TK = f'model/tokenizer-{args.tokenizer}{TK_SUFFIX}'
DISTILL_CONFIG = f"{NEW_TK}/config.json"

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

TEMP = args.temp
LEARNING_RATE = args.lr
ADAM_EPSILON = 1e-8

print(f"{args}\n# of epoch = {EPOCH}, Batch size = {BATCH_SIZE}, Block size = {BLOCK_SIZE}, Learning rate = {LEARNING_RATE}")

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # only supports a single GPU
DEVICE = torch.device("cuda")
ALT_DEVICE = torch.device("cuda:1")
