"""
SETN
    version 0: no transfer. original setn
    version 1: search w/a only by matching_loss
    version 2: search w/a by (ori_loss + matching_loss)
    version 3: training w with matching_loss T=2 times -> training w with ori_loss -> training a with matching_loss T=2 times -> training a with ori_loss
    version 4: training w with matching_loss T=2 times -> training w with ori_loss -> training a with (matching_loss + ori_loss)
"""
import os, sys, time, torch
import torch.nn.functional as F
from log_utils import AverageMeter, time_string
from utils     import obtain_accuracy

def loss_KD_fn(criterion, student_logits, teacher_logits, targets, alpha=0.9, temperature=4, kd_coef=0.):
  # KD training
  basic_loss = criterion(student_logits, targets) * (1. - alpha)
  if kd_coef != 0.:
      log_student= F.log_softmax(student_logits / temperature, dim=1)
      sof_teacher= F.softmax    (teacher_logits / temperature, dim=1)
      KD_loss    = F.kl_div(log_student, sof_teacher, reduction='batchmean') * (alpha * temperature * temperature)
  else:
      KD_loss = 0.
  return basic_loss + kd_coef * KD_loss

def search_func_setn(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger, teacher=None, matching_layers=None, config=None, kd_coef=0.):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, arch_losses = AverageMeter(), AverageMeter()
  base_top1, base_top5 = AverageMeter(), AverageMeter()
  arch_top1, arch_top5 = AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  if teacher:
    teacher.eval()
    matching_layers.train()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)

    # update the weights
    sampled_arch = network.module.dync_genotype(True) # uniform sampling
    network.module.set_cal_mode('dynamic', sampled_arch)
    network.zero_grad()
    if teacher:
        matching_layers.train()
        matching_layers.zero_grad()

    _, logits, st_outs = network(base_inputs, out_all=True)
    if teacher:
        with torch.no_grad():
            _, t_logits, t_outs = teacher(base_inputs, out_all=True)
        matching_loss = matching_layers(t_outs, st_outs)
        base_loss = torch.mean(matching_loss)
    else:
        base_loss = criterion(logits, base_targets)
    base_loss.backward()
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))
    base_losses.update(base_loss.item(),  base_inputs.size(0))

    # update the architecture-weight
    network.module.set_cal_mode( 'joint' )
    network.zero_grad()
    if teacher:
        matching_layers.eval()
        with torch.no_grad():
            _, t_logits, t_outs = teacher(arch_inputs, True)
        _, logits, st_outs = network(arch_inputs, True)
        matching_loss = matching_layers(t_outs, st_outs)
        arch_loss = torch.mean(matching_loss)
    else:
        _, logits = network(arch_inputs)
        arch_loss = criterion(logits, arch_targets)

    arch_loss.backward()
    a_optimizer.step()
    # record
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
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
      #print (nn.functional.softmax(network.module.arch_parameters, dim=-1))
      #print (network.module.arch_parameters)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg

def search_func_setn_v2(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger, teacher, matching_layers, config=None, kd_coef=0.):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, arch_losses = AverageMeter(), AverageMeter()
  base_top1, base_top5 = AverageMeter(), AverageMeter()
  arch_top1, arch_top5 = AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  teacher.eval()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    matching_layers.train()
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)

    # update the weights
    sampled_arch = network.module.dync_genotype(True) # uniform sampling
    network.module.set_cal_mode('dynamic', sampled_arch)
    with torch.no_grad():
        _, t_logits, t_outs = teacher(base_inputs, out_all=True)
    w_optimizer.zero_grad()
    _, logits, st_outs = network(base_inputs, out_all=True)
    matching_loss = matching_layers(t_outs, st_outs)
    base_loss1 = torch.mean(matching_loss)
    if config is not None and hasattr(config, 'KD_alpha') and hasattr(config, 'KD_temperature'):
        base_loss2 = loss_KD_fn(criterion, logits, t_logits, base_targets, config.KD_alpha, config.KD_temperature, kd_coef)
    else:
        base_loss2 = criterion(logits, base_targets)
    base_loss = base_loss1 + base_loss2
    base_loss.backward()
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))
    base_losses.update(base_loss.item(),  base_inputs.size(0))

    # update the architecture-weight
    network.module.set_cal_mode( 'joint' )
    network.zero_grad()
    matching_layers.eval()
    with torch.no_grad():
        _, t_logits, t_outs = teacher(arch_inputs, True)
    a_optimizer.zero_grad()
    _, logits, st_outs = network(arch_inputs, True)
    matching_loss = matching_layers(t_outs, st_outs)
    arch_loss1 = torch.mean(matching_loss)
    if config is not None and hasattr(config, 'KD_alpha') and hasattr(config, 'KD_temperature'):
        arch_loss2 = loss_KD_fn(criterion, logits, t_logits, arch_targets, config.KD_alpha, config.KD_temperature, kd_coef)
    else:
        arch_loss2 = criterion(logits, arch_targets)
    arch_loss = arch_loss1 + arch_loss2
    arch_loss.backward()
    a_optimizer.step()
    # record
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
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})]'.format(loss=base_losses)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})]'.format(loss=arch_losses)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


def search_func_setn_v3(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger, teacher, matching_layers, config=None, kd_coef=0.):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, arch_losses = AverageMeter(), AverageMeter()
  base_top1, base_top5 = AverageMeter(), AverageMeter()
  arch_top1, arch_top5 = AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  teacher.eval()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    matching_layers.train()
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)

    # update the weights
    sampled_arch = network.module.dync_genotype(True) # uniform sampling
    network.module.set_cal_mode('dynamic', sampled_arch)
    with torch.no_grad():
        _, t_logits, t_outs = teacher(base_inputs, out_all=True)
    for _ in range(2):
        w_optimizer.zero_grad()
        _, logits, st_outs = network(base_inputs, out_all=True)
        matching_loss = matching_layers(t_outs, st_outs)
        base_loss1 = torch.mean(matching_loss)
        base_loss1.backward(retain_graph=True)
        w_optimizer.step()
    w_optimizer.zero_grad()
    base_loss2 = criterion(logits, base_targets)
    base_loss2.backward()
    w_optimizer.step()
    with torch.no_grad():
        base_loss = base_loss1 + base_loss2
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))
    base_losses.update(base_loss.item(),  base_inputs.size(0))

    # update the architecture-weight
    network.module.set_cal_mode( 'joint' )
    network.zero_grad()
    matching_layers.eval()
    with torch.no_grad():
        _, t_logits, t_outs = teacher(arch_inputs, True)
    for _ in range(2):
        a_optimizer.zero_grad()
        _, logits, st_outs = network(arch_inputs, True)
        matching_loss = matching_layers(t_outs, st_outs)
        arch_loss1 = torch.mean(matching_loss)
        arch_loss1.backward(retain_graph=True)
        a_optimizer.step()
    a_optimizer.zero_grad()
    arch_loss2 = criterion(logits, arch_targets)
    arch_loss2.backward()
    a_optimizer.step()
    with torch.no_grad():
        arch_loss = arch_loss1 + arch_loss2
    # record
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
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})]'.format(loss=base_losses)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})]'.format(loss=arch_losses)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


def search_func_setn_v4(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger, teacher, matching_layers, config=None, kd_coef=0.):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, arch_losses = AverageMeter(), AverageMeter()
  base_top1, base_top5 = AverageMeter(), AverageMeter()
  arch_top1, arch_top5 = AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  teacher.eval()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    w_optimizer.zero_grad()
    a_optimizer.zero_grad()
    matching_layers.train()
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)

    # update the weights
    sampled_arch = network.module.dync_genotype(True) # uniform sampling
    network.module.set_cal_mode('dynamic', sampled_arch)
    with torch.no_grad():
        _, t_logits, t_outs = teacher(base_inputs, out_all=True)
    for _ in range(2):
        _, logits, st_outs = network(base_inputs, out_all=True)
        matching_loss = matching_layers(t_outs, st_outs)
        base_loss1 = torch.mean(matching_loss)
        base_loss1.backward(retain_graph=True)
        w_optimizer.step()
        w_optimizer.zero_grad()
    base_loss2 = criterion(logits, base_targets)
    base_loss2.backward()
    w_optimizer.step()
    with torch.no_grad():
        base_loss = base_loss1 + base_loss2
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))
    base_losses.update(base_loss.item(),  base_inputs.size(0))

    # update the architecture-weight
    network.module.set_cal_mode( 'joint' )
    network.zero_grad()
    matching_layers.eval()
    with torch.no_grad():
        _, t_logits, t_outs = teacher(arch_inputs, True)
    _, logits, st_outs = network(arch_inputs, True)
    matching_loss = matching_layers(t_outs, st_outs)
    arch_loss1 = torch.mean(matching_loss)
    arch_loss2 = criterion(logits, arch_targets)
    arch_loss = arch_loss1 + arch_loss2
    arch_loss.backward()
    a_optimizer.step()
    # record
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
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})]'.format(loss=base_losses)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})]'.format(loss=arch_losses)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg
