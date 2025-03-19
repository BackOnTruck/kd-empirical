import json

for (fin, fout) in [('Clone-detection-BigCloneBench/dataset/data.jsonl', 'clone.java'),
        ('Defect-detection/dataset/train.jsonl', 'defect.c'),
        ('Exception-classification/train.jsonl', 'except.py')]:
    with open(fin) as f, open(fout, 'w') as g:
        for line in f:
            print(json.loads(line)['func'].strip() + '\n', file=g)
