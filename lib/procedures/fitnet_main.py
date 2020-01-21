##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, time, torch
import torch.nn.functional as F
# our modules
from log_utils import AverageMeter, time_string
from utils     import obtain_accuracy


def fitnet_train(xloader, teacher, network, matching_layers, criterion, scheduler, optimizer, optim_config, extra_info, print_freq, logger, kd_coef=0., version=1):
    return procedure(xloader, teacher, network, matching_layers, criterion, scheduler, optimizer, 'train', optim_config, extra_info, print_freq, logger, \
                                 kd_coef, version)
    return
def fitnet_valid(xloader, teacher, network, matching_layers, criterion, optim_config, extra_info, print_freq, logger, kd_coef=0., version=1):
  with torch.no_grad():
    return procedure(xloader, teacher, network, matching_layers, criterion, None, None, 'valid', optim_config, extra_info, print_freq, logger, \
                                 kd_coef, version)


def loss_KD_fn(criterion, student_logits, teacher_logits, studentFeatures, teacherFeatures, targets, alpha, temperature, kd_coef=0.):
  # Normal + KD training
  basic_loss = criterion(student_logits, targets) * (1. - alpha)
  if kd_coef != 0.:
      log_student= F.log_softmax(student_logits / temperature, dim=1)
      sof_teacher= F.softmax    (teacher_logits / temperature, dim=1)
      KD_loss    = F.kl_div(log_student, sof_teacher, reduction='batchmean') * (alpha * temperature * temperature)
  else:
      KD_loss = 0.
  return basic_loss + kd_coef * KD_loss

def loss_fitnet_fn(teacher_features, student_features, matching_layers):
    # matching layer + guided training
    matching_loss = matching_layers(teacher_features, student_features)
    return torch.mean(matching_loss)

def procedure(xloader, teacher, network, matching_layers, criterion, scheduler, optimizer, mode, config, extra_info, print_freq, logger, \
              kd_coef=0., version=1):
  """
    version 1: training matching_loss only
    version 2: training CLS_loss(+KD_loss) + matching_loss
    version 3: training CLS_loss 2 times -> training matching_loss
  """
  data_time, batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  Ttop1, Ttop5 = AverageMeter(), AverageMeter()
  if version >= 2:
      matching_losses, kd_losses = AverageMeter(), AverageMeter()
  if mode == 'train':
    network.train()
    matching_layers.train()
  elif mode == 'valid':
    network.eval()
    matching_layers.eval()
  else: raise ValueError("The mode is not right : {:}".format(mode))
  teacher.eval()
  # update the weights
  end = time.time()
  for i, (inputs, targets) in enumerate(xloader):
    if mode == 'train': scheduler.update(None, 1.0 * i / len(xloader))
    # measure data loading time
    data_time.update(time.time() - end)
    # calculate prediction and loss
    targets = targets.cuda(non_blocking=True)
    # matching loss
    T = 2 if version == 3 and mode == 'train' else 1
    with torch.no_grad():
        teacher_f, teacher_logits, teacher_features = teacher(inputs, out_all=True)
    if mode == 'train': optimizer.zero_grad()
    for _ in range(T):
        student_f, logits, student_features = network(inputs, out_all=True)
        matching_loss = loss_fitnet_fn(teacher_features, student_features, matching_layers)
        if version == 3 and mode == 'train':
            matching_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
    # cls(+kd) loss
    if version >= 2:
        loss_KD = loss_KD_fn(criterion, logits, teacher_logits, student_f, teacher_f, targets, config.KD_alpha, config.KD_temperature, kd_coef)
    # loss train
    if version == 1:
        loss = matching_loss
    elif version == 2:
        loss = matching_loss + loss_KD
    elif version == 3:
        loss = loss_KD
    else:
        raise NotImplementedError
    # add aux loss
    if isinstance(logits, list):
      assert len(logits) == 2, 'logits must has {:} items instead of {:}'.format(2, len(logits))
      logits, logits_aux = logits
    else:
      logits, logits_aux = logits, None
    if config is not None and logits_aux is not None and hasattr(config, 'auxiliary') and config.auxiliary > 0:
      loss_aux = criterion(logits_aux, targets)
      loss += config.auxiliary * loss_aux

    if mode == 'train':
        loss.backward()
        optimizer.step()

    # record
    if version >= 2:
        matching_losses.update(matching_loss.item(), inputs.size(0))
        kd_losses.update(loss_KD.item(), inputs.size(0))
    # student
    sprec1, sprec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),   inputs.size(0))
    top1.update  (sprec1.item(), inputs.size(0))
    top5.update  (sprec5.item(), inputs.size(0))
    # teacher
    tprec1, tprec5 = obtain_accuracy(teacher_logits.data, targets.data, topk=(1, 5))
    Ttop1.update (tprec1.item(), inputs.size(0))
    Ttop5.update (tprec5.item(), inputs.size(0))
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % print_freq == 0 or (i+1) == len(xloader):
      Sstr = ' HINT {:5s} '.format(mode.upper()) + time_string() + ' [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(xloader))
      if scheduler is not None:
        Sstr += ' {:}'.format(scheduler.get_min_info())
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=losses, top1=top1, top5=top5)
      if version >= 2:
          Lstr += ' Matching Loss {matching_loss.val:.3f} ({matching_loss.avg:.3f}) KD+Cls Loss {kd_loss.val:.3f} ({kd_loss.avg:.3f})'.format(matching_loss=matching_losses, kd_loss=kd_losses)
      Lstr+= ' Teacher : acc@1={:.2f}, acc@5={:.2f}'.format(Ttop1.avg, Ttop5.avg)
      Istr = 'Size={:}'.format(list(inputs.size()))
      logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Istr)

  if version >= 2:
      logger.log(' **HINT {:5s}** accuracy drop :: @1={:.2f}, @5={:.2f}'.format(mode.upper(), Ttop1.avg - top1.avg, Ttop5.avg - top5.avg))
      logger.log(' **{mode:5s}** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Loss:{loss:.3f}'.format(mode=mode.upper(), top1=top1, top5=top5, error1=100-top1.avg, error5=100-top5.avg, loss=losses.avg))
      return losses.avg, top1.avg, top5.avg, matching_losses.avg, kd_losses.avg
  return losses.avg, top1.avg, top5.avg
