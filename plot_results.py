import os, sys, argparse
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
import torch
import numpy as np
import matplotlib.pyplot as plt

def main(checkpoint_path, save_dir):
    checkpoint = torch.load(checkpoint_path)
    print("Checkpoint load from {:}".format(checkpoint_path))
    args = checkpoint['args']
    # epoch = checkpoint['epoch']
    train_results = checkpoint['train_results']
    valid_results = checkpoint['valid_results']
    if len(train_results) == 3:
        (train_losses, train_acc1s, train_acc5s) = train_results
        (valid_losses, valid_acc1s, valid_acc5s) = valid_results
        valid_acc1s = [ valid_acc1s[i] for i in valid_acc1s if isinstance(i, int) ]
        train_results = (train_losses, train_acc1s)
        valid_results = (valid_losses, valid_acc1s)
        transfer = False
        fig, axes = plt.subplots( nrows = 1, ncols = 2, figsize=(4.8*2, 4.8) )
    elif len(train_results) == 5:
        (train_losses, train_acc1s, train_acc5s, train_matching_losses, train_kd_losses) = train_results
        (valid_losses, valid_acc1s, valid_acc5s, valid_matching_losses, valid_kd_losses) = valid_results
        valid_acc1s = [ valid_acc1s[i] for i in valid_acc1s if isinstance(i, int) ]
        train_results = (train_losses, train_acc1s, train_matching_losses, train_kd_losses)
        valid_results = (valid_losses, valid_acc1s, valid_matching_losses, valid_kd_losses)
        transfer = True
        fig, axes = plt.subplots( nrows = 1, ncols = 4, figsize=(4.8*4, 4.8) )
    else:
        raise ValueError('invalid size: len(train_results): {:}, len()')
    metrics = ["Loss", "Accuracy", "Transfer loss", "Ori. Loss"]
    for i, (train_result, valid_result) in enumerate(zip(train_results, valid_results)):
        axes[i].plot(train_result, label="train")
        axes[i].plot(valid_result, label="valid")
        axes[i].set_xlabel("epoch")
        axes[i].set_ylabel(metrics[i])
        axes[i].legend()
        print("Plot {:}".format(metrics[i]))
    plt.tight_layout()
    save_path = os.path.join(save_dir, args.exp_name + "-{}".format(args.rand_seed) + "-v{}".format(args.version) if transfer else "")
    fig.savefig(save_path + ".png")
    print("Save to {:}".format(save_path + ".png"))

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    if len(sys.argv) >= 3:
        save_dir = sys.argv[2]
    else:
        save_dir = "./plots/"
    main(checkpoint_path, save_dir)
