import os, sys, argparse
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
import torch
import numpy as np
import matplotlib.pyplot as plt
from nas_201_api  import NASBench201API as API
def dict2list(data):
    # assert type(data) == dict, "Invalid input type: {}",format(type(data))
    # if "best" in data:
    #     epochs = len(data) - 1
    # else:
    #     epochs = len(data)
    if type(data) == 'list':
        return data
    list_data = []
    for i in data:
        if i == "best": continue
        list_data.append(data[i])
    return list_data

def main(checkpoint_path, save_dir):
    checkpoint = torch.load(checkpoint_path)
    print("Checkpoint load from {:}".format(checkpoint_path))
    api = API('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_0-e61699.pth'))
    print('create API = {:} done'.format(api))
    args = checkpoint['args']
    dataset = args.dataset
    save_path = os.path.join(save_dir, "{}-{}".format(args.exp_name, args.rand_seed))
    if hasattr(args, "version"):
        save_path += "-v{}".format(args.version)
    save_path1 = save_path + "-sampled-result.png"
    save_path2 = save_path + "-search-result.png"

    epoch = checkpoint['epoch']
    genotypes = checkpoint['genotypes']
    test_acc1 = []
    for i in genotypes:
        if type(i) != int: continue
        index = api.query_index_by_arch(genotypes[i])
        assert index != -1, "Invalid genotype: {}".format(genotype)
        test_acc1.append(api.get_more_info(index, dataset)['test-accuracy'])
    # search_losses = checkpoint['search_losses']
    # search_arch_losses = checkpoint['search_arch_losses']
    # valid_losses = checkpoint['valid_losses']
    # valid_acc1s = checkpoint['valid_acc1s']
    results2 = ['search_losses', 'search_arch_losses', 'valid_losses', 'valid_acc1s']

    # plot1: train-from-scratch results
    fig1, ax1 = plt.subplots( nrows = 1, ncols = 1, figsize=(4.8, 4.8) )
    ax1.plot(test_acc1)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel("Test Accuracy")
    plt.tight_layout()
    fig1.savefig(save_path1)
    print("[Plot1] Saved to {:}".format(save_path1))
    # plot2: search results
    fig2, axes2 = plt.subplots( nrows = 2, ncols = 2, figsize=(4.8*2, 4.8) )
    for k, result in enumerate(results2):
        i = k // 2
        j = k % 2
        axes2[i,j].plot(dict2list(checkpoint[result]))
        axes2[i,j].set_xlabel('epoch')
        axes2[i,j].set_ylabel(result)
    plt.tight_layout()
    fig2.savefig(save_path2)
    print("[Plot2] Saved to {:}".format(save_path2))

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    if len(sys.argv) >= 3:
        save_dir = sys.argv[2]
    else:
        save_dir = "./plots/"
    main(checkpoint_path, save_dir)
