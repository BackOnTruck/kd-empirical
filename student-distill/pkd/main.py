from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from params import *
from dataset import TextDataset
from eval import evaluation
from model import Model

def main():
    print(f"======== Training (Model: small) ========")

    if not test_only:
        train_loader = DataLoader(TextDataset(TRAIN_DATASET, False), batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(TextDataset(VALID_DATASET, True), batch_size=BATCH_SIZE)
        max_steps = EPOCH * len(train_loader)

        model = Model(True)
        params = model.parameters()
        model.to(DEVICE)

        optimizer = AdamW(params, lr=LEARNING_RATE, eps=ADAM_EPSILON, weight_decay=0.0)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1, num_training_steps=max_steps)

        best_acc = best_f1 = 0.0
        for iter in tqdm(range(EPOCH), desc=">>> Training"):
            model.train()
            epoch_loss = 0
            for step, batch in enumerate(bar := tqdm(train_loader)):
                labels, inputs, masks, typeids, t_inputs, t_masks, t_ids = [x.to(DEVICE) for x in batch]
                loss: Tensor = model(labels=labels, input_ids=inputs, attention_mask=masks, token_type_ids=typeids,
                    t_inputs=t_inputs, t_masks=t_masks, t_ids=t_ids).loss

                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)

                epoch_loss += loss.item()
                bar.set_description(f"Epoch #{iter + 1}/{EPOCH}: loss = {epoch_loss / (step + 1):.5f}")

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            acc, F1 = evaluation(model, valid_loader)
            if acc + F1 > best_acc + best_f1:
                print(f">>> Saving model (Accuracy: {best_acc:.2f} => {acc:.2f}, F1: {best_f1:.2f} => {F1:.2f})...\n")
                model.save_pretrained(DISTILLED_MODEL)
                best_acc, best_f1 = acc, F1

    model = Model(False)
    test_loader = DataLoader(TextDataset(TEST_DATASET, True), batch_size=BATCH_SIZE)
    acc, F1 = evaluation(model, test_loader)
    with open(f'{DISTILLED_MODEL}/result.log', 'w') as f:
        print(f'{args}\nAcc {acc:.2f} {"F1" if task != "except" else "Top-3"} {F1:.2f}', file=f)

if __name__ == '__main__':
    main()
