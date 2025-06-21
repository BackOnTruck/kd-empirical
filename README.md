# Empirical of Knowledge Distillation

This is the repository for the paper `An Empirical Study of Knowledge Distillation for Code Understanding Tasks`, accepted by ICSE 2026 (Cycle 1).

## Repository Structure

* `vocab`: generates vocabulary and trains tokenizer given dataset
* `teacher-ft`: fine-tunes teacher pre-trained models on downstream tasks
* `student-distill`: distills or fine-tunes student models on downstream tasks
  * `kd-ft`: vanilla knowledge distillation, or standard fine-tuning
  * `pkd`: patience knowledge distillation
  * `dist`: distillation from a stronger teacher
  * `ckd`: contextual knowledge distillation

## Usage

Invoke `main.py` under each directory to perform training or inference.

### Hyperparameters

* `--gpu`: GPU ID; we only support a single GPU for training and inference
* `--task`: defect / clone / except
* `--lr`: learning rate
* `--teacher`: teacher PLM; unixcoder / modernbert
* `--student`: student architecture; roberta / gru
* `--tokenizer`: student vocabulary size; 1000 / 2000 / 5000 / 20000 / 50000
* `--temp`: temperature
* method-dependent hyperparameters

### Examples

* Training teacher PLMs (`teacher-ft`): `python main.py --teacher modernbert --task defect --gpu 0`
* Training student model (`student-distill`):
  * `kd-ft`: `python main.py --task defect --student roberta --tokenizer 5000 --gpu 0 --lr 3e-4 --temp 10 --teacher unixcoder`
    * for standard fine-tuning: `--no_distill`
  * `pkd`: `python main.py --task defect --student roberta --tokenizer 5000 --gpu 0 --lr 1e-4 --beta 10 --temp 10`
  * `dist`: `python main.py --task defect --student gru --tokenizer 20000 --gpu 0 --lr 1e-4 --temp 10`
  * `ckd`: `python main.py --task defect --student roberta --tokenizer 20000 --lr 3e-5 --beta 1 --gamma 1 --temp 3 --gpu 3`
