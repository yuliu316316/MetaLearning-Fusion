from trainer import *
from option import args
from model import *
import os

if __name__ == '__main__':
    MLF_model = MLFusion(args)
    MLF_model, start_epoch, ok = load_ckp(args, MLF_model)

    t = Trainer(args, MLF_model)

    if args.mode == 'train':
        t.train(start_epoch)

    elif ok == True: 
        t.test()

    else:
        print('[!] No checkpoint file, please train first ! ')