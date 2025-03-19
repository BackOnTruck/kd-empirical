import torch, warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning as UMW

from params import *

def evaluation(model: torch.nn.Module, dataloader: DataLoader):
    model.eval()
    model.to(DEVICE)

    preds, labels, all_logits = [], [], []
    for batch in tqdm(dataloader, desc=">>> Evaluating"):
        with torch.no_grad():
            label, inputs, masks, typeids = [x.to(DEVICE) for x in batch[:4]]
            logits = model(input_ids=inputs, attention_mask=masks, token_type_ids=typeids).logits
            pred = torch.argmax(logits, dim=-1)

        all_logits += logits.tolist()
        labels += label.tolist()
        preds += pred.tolist()

    avg = 'macro' if task == 'except' else 'binary'
    acc = accuracy_score(labels, preds) * 100
    F1 = f1_score(labels, preds, average=avg) * 100

    print(f"\nResults:")
    print(f"    Accuracy = {acc:.2f}")
    print(f"    Recall = {recall_score(labels, preds, average=avg) * 100:.2f}")
    print(f"    Precision = {precision_score(labels, preds, average=avg) * 100:.2f}")
    print(f"    F1 = {F1:.2f}")

    if task == 'except':
        correct = 0
        for logit, label in zip(all_logits, labels):
            top3_labels = sorted(range(20), key=lambda k: logit[k], reverse=True)[:3] # 20 types of exceptions
            correct += label in top3_labels

        top_3 = correct / len(labels) * 100
        print(f'    Top-3 = {top_3:.2f}\n')
        return acc, top_3

    print()
    return acc, F1

warnings.filterwarnings('ignore', category=UMW)
