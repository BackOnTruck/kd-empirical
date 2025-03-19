from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', type=int, required=True, choices=(1000, 2000, 5000, 20000, 50000))
parser.add_argument('--task', required=True, choices=('codexglue', 'except'))
args = parser.parse_args()

VOCAB, TASK = args.vocab, args.task
tokenizer = ByteLevelBPETokenizer()

files, folder = {
    'codexglue': (["clone.java", 'defect.c'], f"../model/tokenizer-{VOCAB}"),
    'except': (['except.py'], f'../model/tokenizer-{VOCAB}-exc')
}[TASK]

tokenizer.train(
    files=files,
    vocab_size=VOCAB,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

tokenizer.post_processor = RobertaProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

os.makedirs(folder, exist_ok=True)
tokenizer.save_model(folder)
