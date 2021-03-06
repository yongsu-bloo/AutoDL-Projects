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


def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger, teacher=None, matching_layers=None):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, arch_losses = AverageMeter(), AverageMeter()
  if teacher is None:
      base_top1, base_top5 = AverageMeter(), AverageMeter()
      arch_top1, arch_top5 = AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  if teacher is not None:
    teacher.eval()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    if teacher is not None:
        matching_layers.train()
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)

    # update the weights
    w_optimizer.zero_grad()
    _, logits, st_outs = network(base_inputs, out_all=True)
    if teacher is not None:
        with torch.no_grad():
            _, t_logits, t_outs = teacher(base_inputs, out_all=True)
        matching_loss = matching_layers(t_outs, st_outs)
        base_loss = torch.mean(matching_loss)
    else:
        base_loss = criterion(logits, base_targets)
    base_loss.backward()
    w_optimizer.step()
    # record
    if teacher is None:
        base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
        base_top1.update  (base_prec1.item(), base_inputs.size(0))
        base_top5.update  (base_prec5.item(), base_inputs.size(0))
    base_losses.update(base_loss.item(),  base_inputs.size(0))

    # update the architecture-weight
    a_optimizer.zero_grad()
    _, logits, st_outs = network(arch_inputs, True)
    if teacher is not None:
        matching_layers.eval()
        with torch.no_grad():
            _, t_logits, t_outs = teacher(arch_inputs, True)
        matching_loss = matching_layers(t_outs, st_outs)
        arch_loss = torch.mean(matching_loss)
    else:
        arch_loss = criterion(logits, arch_targets)
    arch_loss.backward()
    a_optimizer.step()
    # record
    if teacher is None:
        arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
        arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
        arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
    arch_losses.update(arch_loss.item(),  arch_inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      if teacher is None:
          Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
          Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      else:
          Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})]'.format(loss=base_losses)
          Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})]'.format(loss=arch_losses)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  if teacher is None:
      return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg
  else:
      return base_losses.avg, arch_losses.avg

def valid_func(xloader, network, criterion):
  data_time, batch_time = AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  with torch.no_grad():
    network.eval()
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      arch_targets = arch_targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - end)
      # prediction
      _, logits = network(arch_inputs)
      arch_loss = criterion(logits, arch_targets)
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
  return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(xargs)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, xargs.cutout_length)
  assert xargs.dataset == 'cifar10', 'currently only support CIFAR-10'
  #config_path = 'configs/nas-benchmark/algos/SETN.config'
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
  # optim_config = load_config(args.optim_config,
  #                             {'class_num': class_num, 'KD_alpha': args.KD_alpha, 'KD_temperature': args.KD_temperature},
  #                             logger)
  matching_optim_config = load_config(xargs.matching_optim_config, {'class_num': class_num}, logger)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  if xargs.model_config is None:
    model_config = dict2config({'name': 'GDAS', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  else:
    model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space'    : search_space,
                                                    'affine'     : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  logger.log('search space : {:}'.format(search_space))
  logger.log('model-config : {:}'.format(model_config))

  search_model = get_cell_based_tiny_net(model_config)
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  #logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space : {:}'.format(search_space))
  if xargs.search_space_name != "nas-bench-201":
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  # Student
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
  # Teacher
  teacher_base = load_net_from_checkpoint(xargs.teacher_checkpoint)
  teacher      = torch.nn.DataParallel(teacher_base).cuda()
  # Matching layer
  matching_layers = FeatureMatching(teacher, network)
  matching_layers.beta = xargs.beta
  matching_layers = torch.nn.DataParallel(matching_layers).cuda()

  h_optimizer, h_scheduler, _ = get_optim_scheduler(search_model.get_weights() + list(matching_layers.parameters()), matching_optim_config)

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
    start_epoch, genotypes = 0, {-1: search_model.genotype(), 'hint': {}}
    search_losses, search_arch_losses = {'hint':{}}, {'hint':{}}
    valid_losses, valid_acc1s, valid_acc5s = {'hint': {'best': float('inf')}}, {'best': -1}, {}

  # hint training
  start_time, transfer_time, epoch_time, total_hint_epoch = time.time(), AverageMeter(), AverageMeter(), matching_optim_config.epochs + matching_optim_config.warmup
  for epoch in range(start_epoch, total_hint_epoch):
    h_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_hint_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_hint_epoch)
    search_model.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_hint_epoch-1) )
    logger.log('\n[Transfer Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(h_scheduler.get_lr())))

    search_h_loss, search_a_loss \
                = search_func(search_loader, network, criterion, h_scheduler, h_optimizer, a_optimizer, epoch_str, xargs.print_freq, logger, teacher, matching_layers)
    transfer_time.update(time.time() - start_time)
    logger.log('[{:}] Transfer search [base] : loss={:.2f}, time-cost={:.1f} s'.format(epoch_str, search_h_loss, transfer_time.sum))
    logger.log('[{:}] Transfer search [arch] : loss={:.2f}'.format(epoch_str, search_a_loss))
    # validation
    valid_a_loss , _ , _  = valid_func(valid_loader, network, criterion)
    # check the best accuracy
    search_losses['hint'][epoch] = search_h_loss
    search_arch_losses['hint'][epoch] = search_a_loss
    valid_losses['hint'][epoch] = valid_a_loss
    genotypes['hint'][epoch] = search_model.genotype()
    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes['hint'][epoch]))
    if search_a_loss < valid_losses['hint']['best']:
      valid_losses['hint']['best'] = search_a_loss
      genotypes['best']            = genotypes['hint'][epoch]
      logger.log('<<<--->>> The {:}-th epoch : find the lowest hint validation loss : {:.2f}%.'.format(epoch_str, search_a_loss))

    with torch.no_grad():
      logger.log('arch-parameters :\n{:}'.format(search_model.show_alphas()))
    if api is not None: logger.log('{:}'.format(api.query_by_arch( genotypes['hint'][epoch] )))
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  for epoch in range(start_epoch, total_epoch):
      w_scheduler.update(epoch, 0.0)
      need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
      epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
      search_model.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_epoch-1) )
      logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

      search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5 \
            = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, logger)
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
                  'args'  : deepcopy(xargs),
                  'search_model': search_model.state_dict(),
                  'w_optimizer' : w_optimizer.state_dict(),
                  'a_optimizer' : a_optimizer.state_dict(),
                  'w_scheduler' : w_scheduler.state_dict(),
                  'genotypes'   : genotypes,
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
  logger.log('Transfer : run {:} epochs, cost {:.1f} s.'.format(total_hint_epoch, transfer_time.sum))
  logger.log('GDAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum, genotypes[total_epoch-1]))
  if api is not None: logger.log('{:}'.format( api.query_by_arch(genotypes[total_epoch-1]) ))
  logger.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser("GDAS-FitNet Search")
  parser.add_argument('--exp_name',           type=str,  default="",   help='Experiment name')
  parser.add_argument('--overwrite',          type=bool, default=False,  help='Overwrite the existing results')
  # Transfer layer
  parser.add_argument('--teacher_checkpoint', type=str,   default="./.latent-data/basemodels/cifar10/ResNet56.pth",          help='The teacher checkpoint in knowledge distillation.')
  parser.add_argument('--matching_optim_config',        type=str,   default="configs/nas-benchmark/algos/transfer-GDAS.config", help='The path of the transfer configuration.')
  parser.add_argument('--beta', type=float, default=2e-2, help='matching loss scale')
  # data
  parser.add_argument('--data_path',          type=str,   default=os.environ['TORCH_HOME'] + "/cifar.python", help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--cutout_length',      type=int,   default=-1,      help='The cutout length, negative means not use.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   default="nas-bench-201", help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   default=4, help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   default=16, help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   default=5, help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   default=1, choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   default="configs/nas-benchmark/algos/GDAS.config", help='The path of the configuration.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--tau_min',            type=float, default=0.1,  help='The minimum tau for Gumbel')
  parser.add_argument('--tau_max',            type=float, default=10,   help='The maximum tau for Gumbel')
  # log
  parser.add_argument('--workers',            type=int,   default=4,    help='number of data loading workers (default: 4)')
  parser.add_argument('--save_dir',           type=str,   default="./output/transfer-search",     help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   default=os.environ['TORCH_HOME'] + "/NAS-Bench-201-v1_0-e61699.pth", help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   default=100, help='print frequency (default: 100)')
  parser.add_argument('--rand_seed',          type=int,   default=-1, help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  if args.exp_name != "":
      args.save_dir = args.save_dir + "/" + args.exp_name
  main(args)
