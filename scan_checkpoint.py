import sys, os, torch
from pathlib import Path
lib_dir = (Path("__file__").parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
# from nas_201_api import NASBench201API as API

def main(checkpoint_path):
    # checkpoint_path = "./output/sample-train/fitnet-cifar10-SETN/uniform/checkpoint/seed-222-best.pth"
    checkpoint = torch.load(checkpoint_path)
    # print(checkpoint.keys())
    # -> dict_keys(['epoch', 'args', 'max_bytes', 'FLOP', 'PARAM', 'model_config', 'genotype', 'nor_train_results', 'optim_config', 'matching_optim_config', 'student_model', 'matching_layers', 'w_scheduler', 'w_optimizer', 'h_scheduler', 'h_optimizer', 'train_hint_losses', 'valid_hint_losses', 'train_losses', 'train_acc1s', 'train_acc5s', 'valid_losses', 'valid_acc1s', 'valid_acc5s'])
    # api = API('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_0-e61699.pth'))
    genotype = checkpoint['genotype']
    # nor_train_results = api.query_by_arch(genotype)
    # print(genotype)
    # print(nor_train_results)
if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    main(checkpoint_path)
