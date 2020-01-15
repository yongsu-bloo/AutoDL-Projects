import os
import random
import numpy as np
import torch

def write_results(args, results):
    results_save_path = './results/result-{}.csv'.format(args.exp_name)
    nor_train_acc1, nor_valid_acc1, train_loss, train_acc1, train_acc5, valid_loss, valid_acc1, valid_acc5 = results

    with open(results_save_path, "r") as f:
        key_names = f.readline().rstrip().split(",")
    with open(results_save_path, 'a+') as res:
        keys = []
        for key in key_names[:-len(results)]:
            keys.append(eval("args.{}".format(key)))
        metrics = [nor_train_acc1, nor_valid_acc1, train_loss, train_acc1, train_acc5, valid_loss, valid_acc1, valid_acc5]
        contents = keys + metrics
        contents = [str(content) if content is not None else "NaN" for content in contents]
        contents= ','.join(contents)
        res.write(contents + '\n')

        # log for total repetitions
        print('Train loss : {:.4f}, Train Accuracy @1 : {:.4f}, Train Accuracy @5 : {:.4f}'.format(
            train_loss,
            train_acc1,
            train_acc5
            ))
        print('Valid loss : {:.4f}, Valid Accuracy @1 : {:.4f}, Valid Accuracy @5 : {:.4f}'.format(
            valid_loss,
            valid_acc1,
            valid_acc5
            ))
