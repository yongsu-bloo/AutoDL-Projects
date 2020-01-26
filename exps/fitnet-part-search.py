##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time, write_results
from models       import get_cell_based_tiny_net, get_search_spaces, load_net_from_checkpoint, FeatureMatching, CellStructure as Structure
from nas_201_api  import NASBench201API as API


def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( args.workers )
  prepare_seed(args.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_path, args.cutout_length)
  assert args.dataset == 'cifar10', 'currently only support CIFAR-10'
  config = load_config(args.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, args.dataset, 'configs/nas-benchmark/', config.batch_size, args.workers)
  # optim_config = load_config(args.optim_config,
  #                             {'class_num': class_num, 'KD_alpha': args.KD_alpha, 'KD_temperature': args.KD_temperature},
  #                             logger)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(args.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(args.dataset, config))

  search_space = get_search_spaces('cell', args.search_space_name)
  if args.model_config is None:
    fixed_genotype = Structure.str2structure( args.fixed_genotype )
    model_config = dict2config({'name': args.nas_name, 'C': args.channel, 'N': args.num_cells,
                                'max_nodes': args.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'affine'   : False, 'track_running_stats': bool(args.track_running_stats),
                                "fixed_genotype"    :   fixed_genotype, "search_position" : args.pos}, None)
  else:
    model_config = load_config(args.model_config, {'num_classes': class_num, 'space'    : search_space,
                                                    'affine'     : False, 'track_running_stats': bool(args.track_running_stats)}, None)
  logger.log('search space : {:}'.format(search_space))
  logger.log('model-config : {:}'.format(model_config))

  search_model = get_cell_based_tiny_net(model_config)
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space : {:}'.format(search_space))
  if args.search_space_name != "nas-bench-201" or args.num_cells != 5:
    api = None
  else:
    api = API(args.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  # Student
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
  # Teacher
  teacher_base = load_net_from_checkpoint(args.teacher_checkpoint)
  teacher      = torch.nn.DataParallel(teacher_base).cuda()
  # Matching layer
  matching_layers = FeatureMatching(teacher, network)
  matching_layers.beta = args.beta
  matching_layers = torch.nn.DataParallel(matching_layers).cuda()

  if last_info.exists() and not xarg.overwrite: # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    search_losses = checkpoint['search_losses']
    valid_losses = checkpoint['valid_losses']
    search_arch_losses = checkpoint['search_arch_losses']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    matching_layers.load_state_dict( checkpoint['matching_layers'] )
    h_optimizer.load_state_dict ( checkpoint['h_optimizer'] )
    h_scheduler.load_state_dict ( checkpoint['h_scheduler'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, genotypes = 0, {-1: search_model.genotype()}
    search_losses, search_arch_losses = {}, {}
    valid_losses, valid_acc1s, valid_acc5s = {}, {'best': -1}, {}

  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  search_func, valid_func = get_search_methods(args.nas_name, args.version)
  for epoch in range(start_epoch, total_epoch):
      w_scheduler.update(epoch, 0.0)
      need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
      epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
      search_model.set_tau( args.tau_max - (args.tau_max-args.tau_min) * epoch / (total_epoch-1) )
      logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

      search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5 \
            = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, args.print_freq, logger, teacher, matching_layers)
      search_time.update(time.time() - start_time)
      logger.log('[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
      logger.log('[{:}] search [arch] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, search_a_loss, search_a_top1, search_a_top5))
      # validation
      valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
      logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
      # check the best accuracy
      search_losses[epoch] = search_w_loss
      search_arch_losses[epoch] = search_a_loss
      valid_losses[epoch] = valid_a_loss
      valid_acc1s[epoch] = valid_a_top1
      valid_acc5s[epoch] = valid_a_top5
      genotypes[epoch] = search_model.genotype()
      logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
      if valid_a_top1 > valid_acc1s['best']:
          valid_acc1s['best'] = valid_a_top1
          genotypes['best']   = genotypes[epoch]
          find_best = True
      else: find_best = False
      # save checkpoint
      save_path = save_checkpoint({'epoch' : epoch + 1,
                  'args'  : deepcopy(args),
                  'search_model': search_model.state_dict(),
                  'w_optimizer' : w_optimizer.state_dict(),
                  'a_optimizer' : a_optimizer.state_dict(),
                  'w_scheduler' : w_scheduler.state_dict(),
                  'genotypes'   : deepcopy(genotypes),
                  'fixed_genotype' : deepcopy(fixed_genotype),
                  "search_position" : args.pos,
                  "search_losses" : deepcopy(search_losses),
                  "search_arch_losses" : deepcopy(search_arch_losses),
                  "valid_losses" : deepcopy(valid_losses),
                  "valid_acc1s" : deepcopy(valid_acc1s),
                  "valid_acc5s" : deepcopy(valid_acc5s),
                  },
                  model_base_path, logger)
      last_info = save_checkpoint({
              'epoch': epoch + 1,
              'args' : deepcopy(args),
              'last_checkpoint': save_path,
              }, logger.path('info'), logger)
      if find_best:
          logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str, valid_a_top1))
          copy_checkpoint(model_base_path, model_best_path, logger)
      with torch.no_grad():
          logger.log('arch-parameters :\n{:}'.format( nn.functional.softmax(search_model.arch_parameters, dim=-1).cpu() ))
      if api is not None: logger.log('{:}'.format(api.query_by_arch( genotypes[epoch] )))
      # measure elapsed time
      epoch_time.update(time.time() - start_time)
      start_time = time.time()

  logger.log('\n' + '-'*100)
  # check the performance from the architecture dataset
  logger.log('{:} : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(args.nas_name, total_epoch, search_time.sum, genotypes[total_epoch-1]))
  if api is not None: logger.log('{:}'.format( api.query_by_arch(genotypes[total_epoch-1]) ))
  logger.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser("FitNet Part Re-Search")
  parser.add_argument('--exp_name',           type=str,  default="",     help='Experiment name')
  parser.add_argument("--nas_name",           type=str,  default="GDAS", help="NAS algorithm to use")
  parser.add_argument("--version",            type=int,  default=0,      help="Search method version")
  parser.add_argument('--overwrite',          type=bool, default=False,  help='Overwrite the existing results')
  # Transfer layer
  parser.add_argument('--teacher_checkpoint', type=str,   default="./.latent-data/basemodels/cifar10/ResNet110.pth",          help='The teacher checkpoint in knowledge distillation.')
  parser.add_argument('--beta',               type=float, default=1.0, help='matching loss scale')
  parser.add_argument("--fixed_genotype",     type=str,   help="Part cell search architecture")
  parser.add_argument("--pos",                type=int,   help="Part cell search stage: [0,1,2]")
  # data
  parser.add_argument('--data_path',          type=str,   default=os.environ['TORCH_HOME'] + "/cifar.python", help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--cutout_length',      type=int,   default=-1,      help='The cutout length, negative means not use.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   default="nas-bench-201", help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   default=4, help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   default=16, help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   default=2, help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   default=1, choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   default="configs/nas-benchmark/algos/transfer-N2-GDAS.config", help='The path of the configuration.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--tau_min',            type=float, default=0.1,  help='The minimum tau for Gumbel')
  parser.add_argument('--tau_max',            type=float, default=10,   help='The maximum tau for Gumbel')
  # log
  parser.add_argument('--workers',            type=int,   default=8,    help='number of data loading workers')
  parser.add_argument('--save_dir',           type=str,   default="./output/transfer-search",     help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   default=os.environ['TORCH_HOME'] + "/NAS-Bench-201-v1_0-e61699.pth", help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   default=100, help='print frequency (default: 100)')
  parser.add_argument('--rand_seed',          type=int,   default=-1, help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  if args.exp_name != "":
      args.save_dir = args.save_dir + "/" + args.exp_name
  main(args)
