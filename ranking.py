##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
######################################################################################
import os, sys, time, glob, random, argparse, pandas as pd
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path("__file__").parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config
from datasets     import get_datasets
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler, get_procedures
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time, write_results
from models       import get_cell_based_tiny_net, get_search_spaces, load_net_from_checkpoint, FeatureMatching, CellStructure as Structure
from nas_201_api  import NASBench201API as API
from collections import OrderedDict
import higher

def get_n_archs(data, n, pick_top=True, order=True):
    """Get top n players by score.
    Returns a dictionary or an `OrderedDict` if `order` is true.
    """
    if pick_top:
        subset = sorted(data.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:n]
    else:
        rand_indicies = random.sample(range(len(data)), n)
        subset = filter(lambda x: x[0] in rand_indicies, data.items())
    if order:
        return OrderedDict(subset)
    else:
        return dict(subset)

def list_arch(api, dataset, metric_on_set, FLOP_max=None, Param_max=None, use_12epochs_result=False):
    """Find the architecture with the highest accuracy based on some constraints."""
    if use_12epochs_result: basestr, arch2infos = '12epochs' , api.arch2infos_less
    else                  : basestr, arch2infos = '200epochs', api.arch2infos_full
    result = OrderedDict()
    for i, arch_id in enumerate(api.evaluated_indexes):
      info = arch2infos[arch_id].get_compute_costs(dataset)
      flop, param, latency = info['flops'], info['params'], info['latency']
      if FLOP_max  is not None and flop  > FLOP_max : continue
      if Param_max is not None and param > Param_max: continue
      xinfo = arch2infos[arch_id].get_metrics(dataset, metric_on_set)
      loss, accuracy = xinfo['loss'], xinfo['accuracy']
      arch_str = api.query_by_index(arch_id).arch_str
      result[arch_id] = { "arch_str": arch_str, "accuracy": accuracy, "flop": flop, "param": param }
    return result

def get_best_arch(xloader, network, n_samples):
  # setn evaluation
  with torch.no_grad():
    network.eval()
    archs, valid_accs = network.return_topK(n_samples), []
    #print ('obtain the top-{:} architectures'.format(n_samples))
    loader_iter = iter(xloader)
    for i, sampled_arch in enumerate(archs):
      network.set_cal_mode('dynamic', sampled_arch)
      try:
        inputs, targets = next(loader_iter)
      except:
        loader_iter = iter(xloader)
        inputs, targets = next(loader_iter)

      _, logits = network(inputs)
      val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))

      valid_accs.append( val_top1.item() )
      #print ('--- {:}/{:} : {:} : {:}'.format(i, len(archs), sampled_arch, val_top1))

    best_idx = np.argmax(valid_accs)
    best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
    return best_arch, best_valid_acc

def valid_func(xloader, network, criterion, print_freq, logger):
  data_time, batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  network.eval()
  end = time.time()
  for i, (inputs, targets) in enumerate(xloader):
    # measure data loading time
    data_time.update(time.time() - end)
    # calculate prediction and loss
    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    features, logits = network(inputs)
    if isinstance(logits, list):
      assert len(logits) == 2, 'logits must has {:} items instead of {:}'.format(2, len(logits))
      logits, logits_aux = logits
    else:
      logits, logits_aux = logits, None
    loss             = criterion(logits, targets)

    # record
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % print_freq == 0 or (i+1) == len(xloader):
      Sstr = time_string() + ' [{:03d}/{:03d}]'.format(i, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=losses, top1=top1, top5=top5)
      Istr = 'Size={:}'.format(list(inputs.size()))
      logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Istr)

  logger.log(' **{mode:5s}** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Loss:{loss:.3f}'.format(mode="test".upper(), top1=top1, top5=top5, error1=100-top1.avg, error5=100-top5.avg, loss=losses.avg))
  return losses.avg, top1.avg, top5.avg


def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  prepare_seed(args.rand_seed)
  checkpoint = torch.load( args.checkpoint )
  xargs      = checkpoint['args']
  exp_name = xargs.exp_name if args.exp_name == "" else args.exp_name
  args.save_dir = "./output/ranking/{}".format(exp_name)
  logger = prepare_logger(args)

  # total_time = time.time()

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, args.data_path, args.cutout_length)
  optim_config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True , num_workers=xargs.workers, pin_memory=True)
  valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=xargs.workers, pin_memory=True)
  logger.log('||||||| {:10s} ||||||| Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(valid_loader), args.batch_size))
  logger.log('||||||| {:10s} ||||||| Optim-Config={:}'.format(xargs.dataset, optim_config))
  search_space = get_search_spaces('cell', xargs.search_space_name)
  model_config = dict2config({'name': xargs.nas_name, 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  logger.log('search space : {:}'.format(search_space))
  logger.log('model-config : {:}'.format(model_config))
  search_model = get_cell_based_tiny_net(model_config)
  flop, param  = get_model_infos(search_model, xshape)
  logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space : {:}'.format(search_space))
  api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))
  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.parameters(), optim_config)
  logger.log('criterion  : {:}'.format(criterion))
  # network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
  search_model.load_state_dict( checkpoint['search_model'] if 'search_model' in checkpoint else checkpoint['shared_cnn'] )
  network, criterion = search_model.cuda(), criterion.cuda()

  n_top = args.n_top
  k_shot = args.k_shot
  # specify search space
  if "search_scope" in checkpoint:
      search_scope = checkpoint['search_scope']
      logger.log("***Search Scope found***")
  elif n_top:
      # all_archs = list_arch(api, xargs.dataset, 'ori-test')
      arch_path = os.environ['TORCH_HOME'] + "/all_archs-{}-test.pt".format(args.dataset)
      if os.path.isfile(arch_path):
          all_archs = torch.load(arch_path)["all_archs"]
      else:
          all_archs = list_arch(api, args.dataset, 'ori-test')
          save_checkpoint({'all_archs': all_archs}, arch_path, logger)
      pick_top = True
      if n_top < 0: # random pick
        n_top = -n_top
        pick_top = False
      assert n_top > 0, "[Picking search space] n_top argument should be int. Now given {} with type {}".format(args.n_top, type(args.n_top))
      search_scope = get_n_archs(all_archs, n_top, pick_top)
      logger.log("***Search scope not found but generated. n_top: {:}, pick_top: {:}***".format(n_top, pick_top))
  else:
      raise ValueError("search_scope is not in checkpoint nor n_top is given properly. Given n_top: {:}".format(n_top))
  # evaluation
  train_loader_iter = iter(train_loader)
  for arch_id in search_scope:
      arch_info = search_scope[arch_id] # -> {arch_str, accuracy, flop, param}
      arch_str = arch_info["arch_str"]
      true_acc1 = arch_info["accuracy"]
      logger.log("-" * 20 + "\nArch id [{:}], Stand-alone test accuracy [{:.2f}], arch_str [ {:} ], flops [{:}], params [{:}]]".format(arch_id, true_acc1, arch_str, arch_info['flop'], arch_info['param']))
      genotype = Structure(API.str2lists(arch_str))
      network.set_cal_mode('dynamic', genotype)
      # @TODO few-shot evaluation
      # training k-step
      with higher.innerloop_ctx(network, w_optimizer, track_higher_grads=False) as (fmodel, diffopt):
          for k in range(k_shot):
              try:
                t_inputs, t_targets = next(train_loader_iter)
              except:
                train_loader_iter = iter(train_loader)
                t_inputs, t_targets = next(train_loader_iter)
              t_inputs = t_inputs.cuda(non_blocking=True)
              t_targets = t_targets.cuda(non_blocking=True)
              # fast gradient
              _, logits = fmodel(t_inputs)
              loss      = criterion(logits, t_targets)
              # torch.nn.utils.clip_grad_norm_(fmodel.parameters(), 5)
              diffopt.step(loss)
          # evaluation
          valid_loss, valid_acc1, valid_acc5 = valid_func(valid_loader, fmodel, criterion, print_freq=xargs.print_freq,logger=logger)

      acc1_gap = true_acc1 - valid_acc1
      logger.log('***{:s}*** EVALUATION loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, AccGap(True_acc1 - Supernet_acc1) = {:.2f}'.format(time_string(), valid_loss, valid_acc1, valid_acc5, acc1_gap))
      search_scope[arch_id]["supernet_acc"] = valid_acc1

  # save result as csv
  result_path = './results/ranking/{}.csv'.format(exp_name)
  # if not os.path.exists(result_path):
  #     os.mkdir(result_path)
  with open(result_path, 'w') as res:
    title =   ["arch_id",
               "arch_str",
               "accuracy",
               "supernet_acc",
               "flops",
               "params"]
    title = ','.join(title)
    res.write(title + '\n')
    for arch_id in search_scope:
        arch_info = search_scope[arch_id] # -> {arch_str, accuracy, supernet_acc , flop, param}
        result = [ str(a) for a in [arch_id, arch_info["arch_str"], arch_info["accuracy"], arch_info["supernet_acc"], arch_info["flop"], arch_info["param"]] ]
        result = ",".join(result)
        res.write(result + '\n')

  res = pd.read_csv(result_path, sep=',', header=0)
  res = res.sort_values('supernet_acc', ascending=False)
  res.to_csv(result_path, sep=',', index=False)

  logger.log('\n' + '-'*100)
  num_bytes = torch.cuda.max_memory_cached( next(network.parameters()).device ) * 1.0
  logger.log('[GPU-Memory-Usage on {:} is {:} bytes, {:.2f} KB, {:.2f} MB, {:.2f} GB.]'.format(next(network.parameters()).device, int(num_bytes), num_bytes / 1e3, num_bytes / 1e6, num_bytes / 1e9))
  logger.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser("Part Re-Search2")
  parser.add_argument('--exp_name',           type=str,   default="",     help='Experiment name')
  parser.add_argument('--overwrite',          type=bool,  default=False,  help='Overwrite the existing results')
  parser.add_argument('--n_top',              type=int,   default=10,     help='The number of top architectures to be scope. If negative, random archs are sampled.')
  parser.add_argument('--k_shot',             type=int,   default=0,      help='The number of training step before the evaluation on test data.')
  # data
  parser.add_argument('--data_path',          type=str,   default=os.environ['TORCH_HOME'] + "/cifar.python", help='Path to dataset')
  parser.add_argument('--checkpoint',         type=str,   help='Checkpoint path')
  parser.add_argument('--batch_size',         type=int,   default=256,     help='Test data mini-batch size')
  parser.add_argument('--cutout_length',      type=int,   default=-1,      help='The cutout length, negative means not use.')
  # log
  parser.add_argument('--save_dir',           type=str,   default="./output/ranking",     help='Folder to save checkpoints and log.')
  # parser.add_argument('--print_freq',         type=int,   default=100, help='print frequency (default: 100)')
  parser.add_argument('--rand_seed',          type=int,   default=-1, help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
  # top 10 archs
  # '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|'
  # '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|'
  # '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
  # '|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|'
  # '|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_1x1~2|'
  # '|nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
  # '|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|'
  # '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
  # '|nor_conv_3x3~0|+|nor_conv_3x3~0|none~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
  # '|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
