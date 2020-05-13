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

def meta_train_shared_cnn(xloader, shared_cnn, criterion, scheduler, optimizer, epoch_str, print_freq, logger, meta_info=(16, 1, 1.)):
  # MAML
  # Sampling: uniform
  data_time, batch_time = AverageMeter(), AverageMeter()
  losses, top1s, top5s, xend = AverageMeter(), AverageMeter(), AverageMeter(), time.time()
  n_task, n_inner_iter, inner_lr_ratio = meta_info
  shared_cnn.train()
  optimizer.zero_grad()
  # for loop: n_iter -> n_task -> data batch
  for step, (qinputs, qtargets, inputs, targets) in enumerate(xloader):
  # loader_iter = iter(xloader)
  # for step in range(n_task):
  #     try:
  #       inputs, targets, qinputs, qtargets = next(loader_iter)
  #     except:
  #       loader_iter = iter(xloader)
  #       inputs, targets, qinputs, qtargets = next(loader_iter)
      scheduler.update(None, 1.0 * step / len(xloader))
      sampled_arch = shared_cnn.dync_genotype(use_random=True)
      shared_cnn.update_arch(sampled_arch)
      inputs = inputs.cuda(non_blocking=True)
      targets = targets.cuda(non_blocking=True)
      qinputs = qinputs.cuda(non_blocking=True)
      qtargets = qtargets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - xend)
      inner_opt = torch.optim.SGD(shared_cnn.parameters(), lr=0.025 * inner_lr_ratio)
      with higher.innerloop_ctx(shared_cnn, inner_opt, copy_initial_weights=False, track_higher_grads=True) as (fmodel, diffopt):
        for _ in range(n_inner_iter):
            _, logits = fmodel(inputs)
            loss      = criterion(logits, targets)
            torch.nn.utils.clip_grad_norm_(fmodel.parameters(), 5)
            diffopt.step(loss)

        _, qlogits = fmodel(qinputs)
        qloss      = criterion(qlogits, qtargets)
        qloss.backward()
        torch.nn.utils.clip_grad_norm_(fmodel.parameters(), 5)

        # record
        prec1, prec5 = obtain_accuracy(qlogits.data, qtargets.data, topk=(1, 5))
        losses.update(qloss.item(), qinputs.size(0))
        top1s.update (prec1.item(), qinputs.size(0))
        top5s.update (prec5.item(), qinputs.size(0))

        if step % n_task == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - xend)
        xend = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
          Sstr = '*Train-Shared-CNN* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
          Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
          Wstr = '[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=losses, top1=top1s, top5=top5s)
          logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  optimizer.step()
  return losses.avg, top1s.avg, top5s.avg

def train_shared_cnn(xloader, shared_cnn, criterion, scheduler, optimizer, epoch_str, print_freq, logger):
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
    shared_cnn.update_arch(sampled_arch)
    shared_cnn.zero_grad()
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


def train_controller(xloader, w_loader, shared_cnn, controller, criterion, optimizer, meta_info, config, epoch_str, print_freq, logger):
  # config. (containing some necessary arg)
  #   baseline: The baseline score (i.e. average val_acc) from the previous epoch
  data_time, batch_time = AverageMeter(), AverageMeter()
  few_shot_time = AverageMeter()
  GradnormMeter, LossMeter, ValAccMeter, EntropyMeter, BaselineMeter, RewardMeter, xend = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()
  (n_shot, inner_lr_ratio) = meta_info
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
    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - xend)

    log_prob, entropy, sampled_arch = controller()
    shared_cnn.update_arch(sampled_arch)
    #few shot start
    if n_shot > 0:
        # shared_cnn.train()
        iloader_iter = iter(w_loader)
        idata_time, ibatch_time = AverageMeter(), AverageMeter()
        losses, top1s, top5s, iend = AverageMeter(), AverageMeter(), AverageMeter(), time.time()
        inner_opt = torch.optim.SGD(shared_cnn.parameters(), lr=0.025 * inner_lr_ratio)
        fs_end = time.time()
        with higher.innerloop_ctx(shared_cnn, inner_opt, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):
            for istep in range(n_shot):
                try:
                  t_inputs, t_targets = next(iloader_iter)
                except:
                  iloader_iter = iter(w_loader)
                  t_inputs, t_targets = next(iloader_iter)
                t_inputs = t_inputs.cuda(non_blocking=True)
                t_targets = t_targets.cuda(non_blocking=True)
                # measure data loading time
                idata_time.update(time.time() - iend)
                # training
                _, logits = fmodel(t_inputs)
                loss      = criterion(logits, t_targets)
                # torch.nn.utils.clip_grad_norm_(fmodel.parameters(), 5)
                diffopt.step(loss)
                # record
                prec1, prec5 = obtain_accuracy(logits.data, t_targets.data, topk=(1, 5))
                losses.update(loss.item(),  t_inputs.size(0))
                top1s.update (prec1.item(), t_inputs.size(0))
                top5s.update (prec5.item(), t_inputs.size(0))
            # measure elapsed time
            ibatch_time.update(time.time() - iend)
            iend = time.time()
            if istep+1 == n_shot:
                Sstr = '--*Few-Shot-Train-Shared-CNN* ' + time_string() + ' [{:03d}|{:02d} steps]'.format(step, istep)
                Tstr = 'Time {ibatch_time.val:.2f} ({ibatch_time.avg:.2f}) Data {idata_time.val:.2f} ({idata_time.avg:.2f})'.format(ibatch_time=ibatch_time, idata_time=idata_time)
                Wstr = '[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=losses, top1=top1s, top5=top5s)
                logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
            few_shot_time.update(time.time() - fs_end)

            # few shot end
            _, logits = fmodel(inputs)
    else:
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

    # measure elapsed time
    batch_time.update(time.time() - xend)
    xend = time.time()

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
  logger = prepare_logger(xargs)

  train_data, test_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  logger.log('use config from : {:}'.format(xargs.config_path))
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, None)
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape, 'LR': config.LR * xargs.lr_ratio, 'eta_min': config.eta_min * xargs.lr_ratio}, logger)
  search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, test_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
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

  supernet_save_path = logger.path('log') / 'seed-{:}-supernet.pth'.format(logger.seed)
  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  if last_info.exists(): # automatically resume from previous checkpoint
    # resume from training controller
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
  elif supernet_save_path.exists():
      # resume from training supernet
      logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
      checkpoint   = torch.load(supernet_save_path)
      start_epoch = checkpoint['epoch']
      # genotypes   = checkpoint['genotypes']
      # baseline    = checkpoint['baseline']
      # valid_accuracies = checkpoint['valid_accuracies']
      valid_accuracies, genotypes, baseline = {'best': -1}, {}, None
      shared_cnn.load_state_dict( checkpoint['shared_cnn'] )
      # controller.load_state_dict( checkpoint['controller'] )
      w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
      w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
      # a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes, baseline = 0, {'best': -1}, {}, None

  # start training supernet
  start_time, supernet_train_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  supernet_train_losses = []
  supernet_train_acc1 = []
  if not xargs.load_supernet:
      meta_info = (xargs.n_task, xargs.n_inner_iter, xargs.inner_lr_ratio)
      for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))
        if 'nometa' in xargs.exp_name:
            cnn_loss, cnn_top1, cnn_top5 = train_shared_cnn(train_loader, shared_cnn, criterion, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, logger)
        else:
            cnn_loss, cnn_top1, cnn_top5 = meta_train_shared_cnn(search_loader, shared_cnn, criterion, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, logger, meta_info=meta_info)
        supernet_train_losses.append(cnn_loss)
        supernet_train_acc1.append(cnn_top1)
        supernet_train_time.update(time.time() - start_time)
        logger.log('[{:}] shared-cnn : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, cnn_loss, cnn_top1, cnn_top5))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        # save checkpoint
        # save_path = save_checkpoint({'epoch' : epoch + 1,
        #             'args'  : deepcopy(xargs),
        #             'baseline'    : baseline,
        #             'shared_cnn'  : shared_cnn.state_dict(),
        #             'controller'  : controller.state_dict(),
        #             'w_optimizer' : w_optimizer.state_dict(),
        #             'a_optimizer' : a_optimizer.state_dict(),
        #             'w_scheduler' : w_scheduler.state_dict(),
        #             'genotypes'   : genotypes,
        #             'valid_accuracies' : valid_accuracies},
        #             model_base_path, logger)
        save_path = save_checkpoint(
                     {'epoch' : epoch + 1,
                      'args'  : deepcopy(xargs),
                      'shared_cnn'  : shared_cnn.state_dict(),
                      'w_optimizer' : w_optimizer.state_dict(),
                      'w_scheduler' : w_scheduler.state_dict()},
                      supernet_save_path, logger)
      logger.log('<<<--->>> Supernet Train Complete.')
  else:
      checkpoint = torch.load(xargs.load_supernet)
      shared_cnn.load_state_dict( checkpoint['shared_cnn'] )
      w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
      w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
  # start search
  start_time, search_time, epoch_time, total_few_shot_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), AverageMeter(), config.search_epochs
  eval_time = AverageMeter()
  meta_info = (xargs.n_shot, xargs.inner_lr_ratio)
  for epoch in range(start_epoch, total_epoch):
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, baseline={:}'.format(epoch_str, need_time, baseline))
    # training controller
    ctl_loss, ctl_acc, ctl_baseline, ctl_reward, baseline, few_shot_time \
                                 = train_controller(valid_loader, train_loader, shared_cnn, controller, criterion, a_optimizer, meta_info, \
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
  parser.add_argument('--exp_name',           type=str,   default="",     help='Experiment name')
  parser.add_argument('--load_supernet',      type=str,   help="supernet path")
  parser.add_argument('--data_path',          type=str,   default=os.environ['TORCH_HOME'] + "/cifar.python", help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--track_running_stats',type=int,   default=0, choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--search_space_name',  type=str,   default="nas-bench-201", help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   default=4,  help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   default=16, help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   default=2,  help='The number of cells in one stage.')
  parser.add_argument('--n_shot',             type=int,   default=1)
  parser.add_argument('--n_task',             type=int,   default=16)
  parser.add_argument('--n_inner_iter',       type=int,   default=1)
  parser.add_argument('--inner_lr_ratio',     type=float, default=1.)
  parser.add_argument('--lr_ratio',           type=float, default=1.)

  parser.add_argument('--config_path',        type=str,   default="./configs/research/MetaENAS250.config", help='The config file to train ENAS.')
  parser.add_argument('--controller_train_steps',    type=int,    default=10,       help='.')
  parser.add_argument('--controller_num_aggregate',  type=int,    default=2,      help='.')
  parser.add_argument('--controller_entropy_weight', type=float,  default=0.0001,  help='The weight for the entropy of the controller.')
  parser.add_argument('--controller_bl_dec'        , type=float,  default=0.99,    help='.')
  parser.add_argument('--controller_num_samples'   , type=int,    default=100,     help='.')
  # log
  parser.add_argument('--workers',            type=int,   default=4,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   default="./output/MetaENAS/", help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   default=os.environ['TORCH_HOME'] + "/NAS-Bench-201-v1_1-096897.pth", help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   default=100, help='print frequency (default: 100)')
  parser.add_argument('--rand_seed',          type=int,   default=-1, help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  if args.exp_name != "":
      args.save_dir = "./output/search-cell-{}/MetaENAS-{}-BN{}/{}".format(args.search_space_name, args.dataset, args.track_running_stats, args.exp_name)
  main(args)
