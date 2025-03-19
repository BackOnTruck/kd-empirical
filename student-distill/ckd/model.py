from params import *

if args.student == 'roberta':
    from model_trans import Model
else:
    from model_gru import Model
