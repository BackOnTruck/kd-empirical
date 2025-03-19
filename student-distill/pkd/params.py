import torch, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", choices=('0', '1', '2', '3'), required=True)
parser.add_argument("--task", choices=('defect', 'clone', 'except'), required=True)
parser.add_argument("--teacher", choices=('codebert', 'unixcoder'), default='unixcoder')
parser.add_argument('--student', choices=('roberta', 'gru'), required=True)
parser.add_argument('--tokenizer', choices=('2000', '5000', '20000'), required=True)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--temp', type=float, default=10.0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=10.0)
parser.add_argument("--test_only", action='store_true')

args = parser.parse_args()
task, teacher, test_only = args.task, args.teacher, args.test_only

ORIGINAL_MODEL = {
    'codebert': 'model/codebert-base',
    'unixcoder': 'model/unixcoder-base'
}[teacher] # pretrained large model

TUNED_MODEL = f"model/{teacher}-tuned-{task}"
DISTILLED_MODEL = f"model/pkd-{task}-{teacher}-{args.student}-{args.tokenizer}-LR{args.lr}-T{args.temp}-A{args.alpha}-B{args.beta}"

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
ALPHA, BETA = args.alpha, args.beta

print(f"{args}\n# of epoch = {EPOCH}, Batch size = {BATCH_SIZE}, Block size = {BLOCK_SIZE}, Learning rate = {LEARNING_RATE}")

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # only supports a single GPU
DEVICE = torch.device("cuda")
ALT_DEVICE = torch.device("cuda:1")
