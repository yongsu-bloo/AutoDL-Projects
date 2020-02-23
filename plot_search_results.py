import os, sys, argparse
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
import torch
import numpy as np
import matplotlib.pyplot as plt
from nas_201_api  import NASBench201API as API

def main(checkpoint_path, save_dir):
    checkpoint = torch.load(checkpoint_path)
    print("Checkpoint load from {:}".format(checkpoint_path))
    api = API(args.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    genotypes = checkpoint['genotypes']

    search_losses = checkpoint['search_losses']
    search_arch_losses = checkpoint['search_arch_losses']

    valid_losses = checkpoint['valid_losses']
    valid_acc1s = checkpoint['valid_acc1s']
    
    metrics = ["Loss", "Accuracy", "Transfer loss", "Ori. Loss"]
    for i, (train_result, valid_result) in enumerate(zip(train_results, valid_results)):
        axes[i].plot(train_result, label="train")
        axes[i].plot(valid_result, label="valid")
        axes[i].set_xlabel("epoch")
        axes[i].set_ylabel(metrics[i])
        axes[i].legend()
        print("Plot {:}".format(metrics[i]))
    plt.tight_layout()
    save_path = os.path.join(save_dir, args.exp_name + "-{}".format(args.rand_seed))
    save_path += "-v{}".format(args.version) if transfer else ""
    fig.savefig(save_path + ".png")
    print("Save to {:}".format(save_path + ".png"))

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    if len(sys.argv) >= 3:
        save_dir = sys.argv[2]
    else:
        save_dir = "./plots/"
    main(checkpoint_path, save_dir)
