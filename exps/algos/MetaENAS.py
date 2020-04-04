##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##########################################################################
# Efficient Neural Architecture Search via Parameters Sharing, ICML 2018 #
##########################################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_201_api  import NASBench201API as API
import higher

def train_shared_cnn(xloader, shared_cnn, criterion, scheduler, optimizer, epoch_str, print_freq, logger):
  # MAML
  # Sampling: uniform
  data_time, batch_time = AverageMeter(), AverageMeter()
  losses, top1s, top5s, xend = AverageMeter(), AverageMeter(), AverageMeter(), time.time()

  shared_cnn.train()

  for step, (inputs, targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    inputs = inputs.cuda()
    targets = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - xend)

    sampled_arch = shared_cnn.dync_genotype(use_random=True)
    optimizer.zero_grad()
    shared_cnn.update_arch(sampled_arch)
    _, logits = shared_cnn(inputs)
    loss      = criterion(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(), 5)
    optimizer.step()
    # record
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1s.update (prec1.item(), inputs.size(0))
    top5s.update (prec5.item(), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - xend)
    xend = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*Train-Shared-CNN* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = '[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=losses, top1=top1s, top5=top5s)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  return losses.avg, top1s.avg, top5s.avg


def few_shot_train_shared_cnn(xloader, shared_cnn, criterion, n_shot, optimizer, print_freq, current_step, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  losses, top1s, top5s, xend = AverageMeter(), AverageMeter(), AverageMeter(), time.time()
  shared_cnn.train()
  loader_iter = iter(xloader)
  for step in range(n_shot):
    try:
      inputs, targets = next(loader_iter)
    except:
      loader_iter = iter(xloader)
      inputs, targets = next(loader_iter)
    inputs = inputs.cuda()
    targets = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - xend)

    # training
    optimizer.zero_grad()
    _, logits = shared_cnn(inputs)
    loss      = criterion(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(), 5)
    optimizer.step()

    # record
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1s.update (prec1.item(), inputs.size(0))
    top5s.update (prec5.item(), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - xend)
    xend = time.time()
    if (step+1) % print_freq == 0 or step+1 == n_shot:
        Sstr = '--*Few-Shot-Train-Shared-CNN* ' + time_string() + ' [{:03d}|{:02d} steps]'.format(current_step, step)
        Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
        Wstr = '[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=losses, top1=top1s, top5=top5s)
        logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  return losses.avg, top1s.avg, top5s.avg

def train_controller(xloader, w_loader, shared_cnn, controller, criterion, optimizer, w_optimizer, n_shot, config, epoch_str, print_freq, logger):
  # config. (containing some necessary arg)
  #   baseline: The baseline score (i.e. average val_acc) from the previous epoch
  data_time, batch_time = AverageMeter(), AverageMeter()
  few_shot_time = AverageMeter()
  GradnormMeter, LossMeter, ValAccMeter, EntropyMeter, BaselineMeter, RewardMeter, xend = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()

  controller.train()
  controller.zero_grad()
  #for step, (inputs, targets) in enumerate(xloader):
  loader_iter = iter(xloader)
  for step in range(config.ctl_train_steps * config.ctl_num_aggre):
    try:
      inputs, targets = next(loader_iter)
    except:
      loader_iter = iter(xloader)
      inputs, targets = next(loader_iter)
    inputs = inputs.cuda()
    targets = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - xend)

    log_prob, entropy, sampled_arch = controller()
    shared_cnn.update_arch(sampled_arch)
    fs_end = time.time()
    few_shot_train_shared_cnn(w_loader, shared_cnn, criterion, n_shot, w_optimizer, print_freq, step, logger)
    few_shot_time.update(time.time() - fs_end)
    with torch.no_grad():
      # few shot train shared_cnn
      shared_cnn.eval()
      _, logits = shared_cnn(inputs)
      val_top1, val_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
      val_top1  = val_top1.view(-1) / 100
    reward = val_top1 + config.ctl_entropy_w * entropy
    if config.baseline is None:
      baseline = val_top1
    else:
      baseline = config.baseline - (1 - config.ctl_bl_dec) * (config.baseline - reward)

    loss = -1 * log_prob * (reward - baseline)

    # account
    RewardMeter.update(reward.item())
    BaselineMeter.update(baseline.item())
    ValAccMeter.update(val_top1.item()*100)
    LossMeter.update(loss.item())
    EntropyMeter.update(entropy.item())

    # Average gradient over controller_num_aggregate samples
    loss = loss / config.ctl_num_aggre
    loss.backward(retain_graph=True)

    # measure elapsed time
    batch_time.update(time.time() - xend)
    xend = time.time()
    if (step+1) % config.ctl_num_aggre == 0:
      grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
      GradnormMeter.update(grad_norm)
      optimizer.step()
      controller.zero_grad()

    if step % print_freq == 0:
      Sstr = '*Train-Controller* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, config.ctl_train_steps * config.ctl_num_aggre)
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f}) Few-shot {few_shot_time.val:.2f} ({few_shot_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time, few_shot_time=few_shot_time)
      Wstr = '[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Reward {reward.val:.2f} ({reward.avg:.2f})] Baseline {basel.val:.2f} ({basel.avg:.2f})'.format(loss=LossMeter, top1=ValAccMeter, reward=RewardMeter, basel=BaselineMeter)
      Estr = 'Entropy={:.4f} ({:.4f})'.format(EntropyMeter.val, EntropyMeter.avg)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Estr)

  return LossMeter.avg, ValAccMeter.avg, BaselineMeter.avg, RewardMeter.avg, baseline.item(), few_shot_time.sum


def get_best_arch(controller, shared_cnn, xloader, n_samples=10):
  with torch.no_grad():
    controller.eval()
    shared_cnn.eval()
    archs, valid_accs = [], []
    loader_iter = iter(xloader)
    for i in range(n_samples):
      try:
        inputs, targets = next(loader_iter)
      except:
        loader_iter = iter(xloader)
        inputs, targets = next(loader_iter)

      _, _, sampled_arch = controller()
      arch = shared_cnn.update_arch(sampled_arch)
      inputs = inputs.cuda()
      _, logits = shared_cnn(inputs)
      val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))

      archs.append( arch )
      valid_accs.append( val_top1.item() )

    best_idx = np.argmax(valid_accs)
    best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
    return best_arch, best_valid_acc

def get_true_best_arch(controller, shared_cnn, xloader, n_samples=10):
  with torch.no_grad():
    controller.eval()
    shared_cnn.eval()
    archs, valid_accs = [], []
    # loader_iter = iter(xloader)
    for i in range(n_samples):
        _, _, sampled_arch = controller()
        arch = shared_cnn.update_arch(sampled_arch)
        valid_top1_mean = 0.
        # valid_top5_mean = 0.
        for inputs, targets in xloader:
          inputs = inputs.cuda()
          _, logits = shared_cnn(inputs)
          val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
          valid_top1_mean += val_top1.item() / len(xloader)
          # valid_top5_mean += val_top5.item() / len(xloader)
        archs.append( arch )
        valid_accs.append( valid_top1_mean )

    best_idx = np.argmax(valid_accs)
    best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
    return best_arch, best_valid_acc

def valid_func(xloader, network, criterion):
  data_time, batch_time = AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.eval()
  end = time.time()
  with torch.no_grad():
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      arch_targets = arch_targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - end)
      # prediction
      arch_inputs = arch_inputs.cuda()
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
  logger = prepare_logger(args)

  train_data, test_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  logger.log('use config from : {:}'.format(xargs.config_path))
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  _, train_loader, valid_loader = get_nas_search_loaders(train_data, test_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
  # since ENAS will train the controller on valid-loader, we need to use train transformation for valid-loader
  valid_loader.dataset.transform = deepcopy(train_loader.dataset.transform)
  if hasattr(valid_loader.dataset, 'transforms'):
    valid_loader.dataset.transforms = deepcopy(train_loader.dataset.transforms)
  # data loader
  logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  model_config = dict2config({'name': 'MetaENAS', 'C': xargs.channel, 'N': xargs.num_cells,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space'    : search_space,
                              'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  shared_cnn = get_cell_based_tiny_net(model_config)
  controller = shared_cnn.create_controller()

  w_optimizer, w_scheduler, criterion = get_optim_scheduler(shared_cnn.parameters(), config)
  a_optimizer = torch.optim.Adam(controller.parameters(), lr=config.controller_lr, betas=config.controller_betas, eps=config.controller_eps)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  #flop, param  = get_model_infos(shared_cnn, xshape)
  #logger.log('{:}'.format(shared_cnn))
  #logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space : {:}'.format(search_space))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))
  shared_cnn, controller, criterion = shared_cnn.cuda(), controller.cuda(), criterion.cuda()

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    baseline    = checkpoint['baseline']
    valid_accuracies = checkpoint['valid_accuracies']
    shared_cnn.load_state_dict( checkpoint['shared_cnn'] )
    controller.load_state_dict( checkpoint['controller'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes, baseline = 0, {'best': -1}, {}, None

  # start training supernet
  start_time, supernet_train_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  supernet_train_losses = []
  supernet_train_acc1 = []
  if not xargs.load_supernet:
      for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

        cnn_loss, cnn_top1, cnn_top5 = train_shared_cnn(train_loader, shared_cnn, criterion, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, logger)
        supernet_train_losses.append(cnn_loss)
        supernet_train_acc1.append(cnn_top1)
        supernet_train_time.update(time.time() - start_time)
        logger.log('[{:}] shared-cnn : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, cnn_loss, cnn_top1, cnn_top5))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
      # save checkpoint
      logger.log('<<<--->>> Supernet Train Complete.')
      supernet_save_path = logger.path('log') / 'seed-{:}-supernet.pth'.format(logger.seed)
      save_path = save_checkpoint(
                 {'epoch' : epoch + 1,
                  'args'  : deepcopy(xargs),
                  'shared_cnn'  : shared_cnn.state_dict(),
                  'w_optimizer' : w_optimizer.state_dict(),
                  'w_scheduler' : w_scheduler.state_dict(),
                  'valid_accuracies' : valid_accuracies},
                  supernet_save_path, logger)

  # start search
  start_time, search_time, epoch_time, total_few_shot_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), AverageMeter(), config.search_epochs
  eval_time = AverageMeter()
  if xargs.load_supernet:
      supernet_load_path = xargs.load_supernet
  else:
      supernet_load_path = supernet_save_path
  checkpoint = torch.load(supernet_load_path)
  n_shot = xargs.n_shot
  for epoch in range(start_epoch, total_epoch):
    # del shared_cnn
    shared_cnn.load_state_dict( checkpoint['shared_cnn'] )
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, baseline={:}'.format(epoch_str, need_time, baseline))
    # training controller
    ctl_loss, ctl_acc, ctl_baseline, ctl_reward, baseline, few_shot_time \
                                 = train_controller(valid_loader, train_loader, shared_cnn, controller, criterion, a_optimizer, w_optimizer, n_shot, \
                                                        dict2config({'baseline': baseline,
                                                                     'ctl_train_steps': xargs.controller_train_steps, 'ctl_num_aggre': xargs.controller_num_aggregate,
                                                                     'ctl_entropy_w': xargs.controller_entropy_weight,
                                                                     'ctl_bl_dec'   : xargs.controller_bl_dec}, None), \
                                                        epoch_str, xargs.print_freq, logger)

    search_time.update(time.time() - start_time)
    total_few_shot_time.update(few_shot_time)
    eval_start = time.time()
    best_arch, best_valid_acc = get_best_arch(controller, shared_cnn, valid_loader)
    eval_time.update(time.time() - eval_start)
    logger.log('[{:}] controller : loss={:.2f}, accuracy={:.2f}%, baseline={:.2f}, reward={:.2f}, current-baseline={:.4f}, search-time={:.1f} s, eval-time={:.1f} s, few-shot-time={:.1f} s'.format(epoch_str, ctl_loss, ctl_acc, ctl_baseline, ctl_reward, baseline, search_time.sum, eval_time.sum, total_few_shot_time.sum))
    genotypes[epoch] = best_arch
    # check the best accuracy
    valid_accuracies[epoch] = best_valid_acc
    if best_valid_acc > valid_accuracies['best']:
      valid_accuracies['best'] = best_valid_acc
      genotypes['best']        = best_arch
      find_best = True
    else: find_best = False

    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'baseline'    : baseline,
                'shared_cnn'  : shared_cnn.state_dict(),
                'controller'  : controller.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict(),
                'genotypes'   : genotypes,
                'valid_accuracies' : valid_accuracies},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)
    if find_best:
      logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str, best_valid_acc))
      copy_checkpoint(model_base_path, model_best_path, logger)
    if api is not None: logger.log('{:}'.format(api.query_by_arch( genotypes[epoch] )))
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()


  logger.log('\n' + '-'*100)
  logger.log('During searching, the best architecture is {:}'.format(genotypes['best']))
  logger.log('Its accuracy is {:.2f}%'.format(valid_accuracies['best']))
  logger.log('Randomly select {:} architectures and select the best.'.format(xargs.controller_num_samples))
  start_time = time.time()
  final_arch, _ = get_best_arch(controller, shared_cnn, valid_loader, xargs.controller_num_samples)
  search_time.update(time.time() - start_time)
  shared_cnn.update_arch(final_arch)
  final_loss, final_top1, final_top5 = valid_func(valid_loader, shared_cnn, criterion)
  logger.log('The Selected Final Architecture : {:}'.format(final_arch))
  logger.log('Loss={:.3f}, Accuracy@1={:.2f}%, Accuracy@5={:.2f}%'.format(final_loss, final_top1, final_top5))
  logger.log('MetaENAS : run {:} epochs, last-geno is {:}.'.format(total_epoch, final_arch))
  if api is not None: logger.log('{:}'.format( api.query_by_arch(final_arch) ))
  logger.log('Time costs: supernet_train_time {:.1f}, search_time {:.1f} (few_shot_time {:.1f}), eval_time {:.1f}'.format(supernet_train_time.sum, search_time.sum, total_few_shot_time.sum, eval_time.sum))
  logger.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser("MetaENAS")
  parser.add_argument('--exp_name',           type=str,  default="",     help='Experiment name')
  parser.add_argument('--load_supernet',      type=str,   help="supernet path")
  parser.add_argument('--data_path',          type=str,   default=os.environ['TORCH_HOME'] + "/cifar.python", help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--track_running_stats',type=int,   default=0, choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--search_space_name',  type=str,   default="nas-bench-201", help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   default=4, help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   default=16, help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   default=5, help='The number of cells in one stage.')
  parser.add_argument('--n_shot',             type=int,   default=5, help='The number of few-shot training step.')
  parser.add_argument('--config_path',        type=str,   default="./configs/research/MetaENAS.config", help='The config file to train ENAS.')
  parser.add_argument('--controller_train_steps',    default=50, type=int,     help='.')
  parser.add_argument('--controller_num_aggregate',  default=20, type=int,     help='.')
  parser.add_argument('--controller_entropy_weight', default=0.0001, type=float,   help='The weight for the entropy of the controller.')
  parser.add_argument('--controller_bl_dec'        , default=0.99, type=float,   help='.')
  parser.add_argument('--controller_num_samples'   , default=100, type=int,     help='.')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   default="./output/MetaENAS/", help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   default=os.environ['TORCH_HOME'] + "/NAS-Bench-201-v1_0-e61699.pth", help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   default=100, help='print frequency (default: 100)')
  parser.add_argument('--rand_seed',          type=int,   default=-1, help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  if args.exp_name != "":
      args.save_dir = "./output/search-cell-{}/MetaENAS-{}-BN{}/{}".format(args. search_space_name, args.dataset, args.track_running_stats, args.exp_name)
  main(args)
