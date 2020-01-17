##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import sys, time, torch, random, argparse, os
from PIL     import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy    import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config #, obtain_cls_fitnet_args as obtain_args
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from procedures   import get_optim_scheduler, get_procedures
from datasets     import get_datasets
from models       import load_net_from_checkpoint, get_cell_based_tiny_net, get_search_spaces, FeatureMatching, CellStructure as Structure
from models       import get_cifar_models, get_imagenet_models
from utils        import get_model_infos
from log_utils    import AverageMeter, time_string, convert_secs2time, write_results
from nas_201_api  import NASBench201API as API

def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False # True
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( args.workers )

  prepare_seed(args.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_path, args.cutout_length)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True , num_workers=args.workers, pin_memory=True)
  valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(args.dataset, len(train_loader), len(valid_loader), args.batch_size))

  # get configures
  optim_config = load_config(args.optim_config,
                                {'class_num': class_num, 'KD_alpha': args.KD_alpha, 'KD_temperature': args.KD_temperature},
                                logger)
  matching_optim_config = load_config(args.matching_optim_config, {'class_num': class_num}, logger)

  #Student
  if args.student_config is None:
      search_space = get_search_spaces('cell', args.search_space_name)
      student_config = dict2config({'name': 'SETN', 'C': args.channel, 'N': args.num_cells,
                                  'max_nodes': args.max_nodes, 'num_classes': class_num,
                                  'space'    : search_space,
                                  'affine'   : False, 'track_running_stats': bool(args.track_running_stats)}, None)
      logger.log('search space : {:}'.format(search_space))
      if args.sample_method != "uniform" and args.arch_str is None:
          # get trained model for architecture sampling
          search_model = get_cell_based_tiny_net(student_config)
          student_checkpoint  = torch.load(args.student_checkpoint)
          search_model.load_state_dict( student_checkpoint['search_model'] )

      student_model = get_cell_based_tiny_net(student_config)
  else:
      student_config = load_config(args.student_config)
      if "cifar" in args.dataset:
          student_model = get_cifar_models(student_config)
      else:
          student_model = get_imagenet_models(student_config)
  network = torch.nn.DataParallel(student_model).cuda()

  # Teacher
  teacher_base = load_net_from_checkpoint(args.teacher_checkpoint)
  teacher      = torch.nn.DataParallel(teacher_base).cuda()
  # Matching layer
  matching_layers = FeatureMatching(teacher, network)
  matching_layers.beta = args.beta
  matching_layers = torch.nn.DataParallel(matching_layers).cuda()

  flop, param  = get_model_infos(student_model, xshape)
  student_weights = student_model.get_weights() if args.student_config is None else list(student_model.parameters())
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(student_weights, optim_config)
  h_optimizer, h_scheduler, _ = get_optim_scheduler(student_weights + list(matching_layers.parameters()), matching_optim_config)

  criterion = criterion.cuda()
  # logger.log('Teacher ====>>>>:\n{:}'.format(teacher_base))
  # logger.log('Student ====>>>>:\n{:}'.format(student_model))
  logger.log('model information : {:}'.format(student_model.get_message()))
  logger.log('-'*50)
  logger.log('Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G'.format(param, flop, flop/1e3))
  logger.log('-'*50)
  logger.log('train_data : {:}'.format(train_data))
  logger.log('valid_data : {:}'.format(valid_data))
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  logger.log('h-optimizer : {:}'.format(h_optimizer))
  logger.log('h-scheduler : {:}'.format(h_scheduler))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  if last_info.exists() and not args.overwrite: # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch'] + 1
    checkpoint  = torch.load(last_info['last_checkpoint'])
    student_model.load_state_dict( checkpoint['student_model'] )
    if args.student_config is None and args.search_space_name == "nas-bench-201":
        genotype = Structure.str2structure( checkpoint['genotype'] )
        nor_train_results = checkpoint['nor_train_results']
        (arch_train_result, arch_valid_result) = nor_train_results
    matching_layers.load_state_dict( checkpoint['matching_layers'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    h_scheduler.load_state_dict ( checkpoint['h_scheduler'] )
    h_optimizer.load_state_dict ( checkpoint['h_optimizer'] )
    max_bytes        = checkpoint['max_bytes']
    hint_results = checkpoint['hint_results']
    train_results = checkpoint['train_results']
    valid_results = checkpoint['valid_results']
    (train_hint_losses, valid_hint_losses) = hint_results
    (train_losses, train_acc1s, train_acc5s) = train_results
    (valid_losses, valid_acc1s, valid_acc5s) = valid_results
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  elif args.resume is not None:
    assert Path(args.resume).exists(), 'Can not find the resume file : {:}'.format(args.resume)
    checkpoint  = torch.load( args.resume )
    start_epoch = checkpoint['epoch'] + 1
    student_model.load_state_dict( checkpoint['student_model'] )
    if args.student_config is None and args.search_space_name == "nas-bench-201":
        genotype = Structure.str2structure( checkpoint['genotype'] )
        nor_train_results = checkpoint['nor_train_results']
        (arch_train_result, arch_valid_result) = nor_train_results
    matching_layers.load_state_dict( checkpoint['matching_layers'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    h_scheduler.load_state_dict ( checkpoint['h_scheduler'] )
    h_optimizer.load_state_dict ( checkpoint['h_optimizer'] )
    max_bytes        = checkpoint['max_bytes']
    hint_results = checkpoint['hint_results']
    train_results = checkpoint['train_results']
    valid_results = checkpoint['valid_results']
    (train_hint_losses, valid_hint_losses) = hint_results
    (train_losses, train_acc1s, train_acc5s) = train_results
    (valid_losses, valid_acc1s, valid_acc5s) = valid_results
    logger.log("=> loading checkpoint from '{:}' start with {:}-th epoch.".format(args.resume, start_epoch))
  elif args.init_model is not None:
    assert Path(args.init_model).exists(), 'Can not find the initialization file : {:}'.format(args.init_model)
    checkpoint  = torch.load( args.init_model )
    student_model.load_state_dict( checkpoint['student_model'] )
    if args.student_config is None:
        genotype = Structure.str2structure( checkpoint['genotype'] )
        nor_train_results = checkpoint['nor_train_results']
        (arch_train_result, arch_valid_result) = nor_train_results
    matching_layers.load_state_dict( checkpoint['matching_layers'] )
    start_epoch = 0
    train_hint_losses = []
    valid_hint_losses = {'best': float('inf')}
    train_losses = []
    train_acc1s = []
    train_acc5s = []
    valid_losses = []
    valid_acc1s = {'best': -1}
    valid_acc5s = []
    logger.log('=> initialize the model from {:}'.format( args.init_model ))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    # genotype sampling
    if args.student_config is None:
        if args.arch_str is None:
            if args.sample_method == 'best':
                genotype = search_model.genotype() # get the best arch
            elif args.sample_method == 'infer':
                genotype = search_model.dync_genotype(use_random=False) # sample arch depending on alphas
            elif args.sample_method == 'uniform':
                genotype = student_model.dync_genotype(use_random=True) # sample uniformly
            else:
                raise ValueError('invalid sample_method: {:}'.format(args.sample_method))
        else:
            genotype = Structure.str2structure(args.arch_str)

        network.module.set_cal_mode('dynamic', genotype)

        if args.search_space_name == "nas-bench-201":
            # Normal Training result from nas201
            api = API('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_0-e61699.pth'))
            logger.log('{:} create API = {:} done'.format(time_string(), api))
            # genotype=Structure.str2structure("|skip_connect~0|+|nor_conv_1x1~0|skip_connect~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|")
            logger.log(genotype)
            arch_index = api.query_index_by_arch(genotype)
            archRes = api.query_meta_info_by_index(arch_index)
            arch_train_result = archRes.get_metrics('cifar10-valid' if args.dataset == "cifar10" else args.dataset, 'train', None,  True)
            arch_valid_result = archRes.get_metrics('cifar10-valid' if args.dataset == "cifar10" else args.dataset, 'x-valid', None,  True)
            nor_train_results = (arch_train_result, arch_valid_result)
            logger.log("-"*100 + "\nNormal Training Result: \n Train : {:}\n Valid : {:}\n".format(*nor_train_results) + "-"*100)

    start_epoch, max_bytes = 0, {}
    train_hint_losses = []
    valid_hint_losses = {'best': float('inf')}
    train_losses = []
    train_acc1s = []
    train_acc5s = []
    valid_losses = []
    valid_acc1s = {'best': -1}
    valid_acc5s = []


  """Main Training and Evaluation Loop"""
  # Hint Training First
  start_time, hint_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), matching_optim_config.epochs + matching_optim_config.warmup
  if args.exp_name == "tmp": total_epoch = 2
  fitnet_train_func, fitnet_valid_func = get_procedures("fitnet")
  for epoch in range(start_epoch, total_epoch):
    h_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.avg * (total_epoch-epoch), True) )
    epoch_str = 'epoch={:03d}/{:03d}'.format(epoch, total_epoch)
    LRs       = h_scheduler.get_lr()
    find_best = False

    logger.log('\n***{:s}*** start {:s} {:s}, LR=[{:.6f} ~ {:.6f}], h_scheduler={:}'.format(time_string(), epoch_str, need_time, min(LRs), max(LRs), h_scheduler))

    # train for one epoch
    train_loss, _, _ = fitnet_train_func(train_loader, teacher, network, matching_layers, criterion, h_scheduler, h_optimizer, matching_optim_config, epoch_str, args.print_freq, logger)
    hint_time.update(time.time() - start_time)
    train_hint_losses.append(train_loss)
    # log the results
    logger.log('***{:s}*** HINT TRAIN [{:}] loss = {:.6f} time-cost = {:.1f}'.format(time_string(), epoch_str, train_loss, hint_time.sum))
    # measure elapsed time
    epoch_time.update(time.time() - start_time)

    # evaluate the performance
    if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
      logger.log('-'*150)
      valid_loss, _, _ = fitnet_valid_func(valid_loader, teacher, network, matching_layers, criterion, optim_config, epoch_str, args.print_freq_eval, logger)
      valid_hint_losses[epoch] = valid_loss
      logger.log('***{:s}*** VALID [{:}] loss = {:.6f} | Best-Loss={:.6f}'.format(time_string(), epoch_str, valid_loss, valid_hint_losses['best']))
      if valid_loss <= valid_hint_losses['best']:
        valid_hint_losses['best'] = valid_loss
        find_best                = True
        logger.log('Currently, the best validation hint loss found at {:03d}-epoch :: loss={:.6f}'.format(epoch, valid_loss))
      num_bytes = torch.cuda.max_memory_cached( next(network.parameters()).device ) * 1.0
      logger.log('[GPU-Memory-Usage on {:} is {:} bytes, {:.2f} KB, {:.2f} MB, {:.2f} GB.]'.format(next(network.parameters()).device, int(num_bytes), num_bytes / 1e3, num_bytes / 1e6, num_bytes / 1e9))
      max_bytes[epoch] = num_bytes
    if epoch % 10 == 0: torch.cuda.empty_cache()
    start_time = time.time()

  # Guided layer training
  start_time, train_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), optim_config.epochs + optim_config.warmup
  if args.exp_name == "tmp": total_epoch = 2
  train_func, valid_func = get_procedures(args.procedure)
  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.avg * (total_epoch-epoch), True) )
    epoch_str = 'epoch={:03d}/{:03d}'.format(epoch, total_epoch)
    LRs       = w_scheduler.get_lr()
    find_best = False

    logger.log('\n***{:s}*** start {:s} {:s}, LR=[{:.6f} ~ {:.6f}], w_scheduler={:}'.format(time_string(), epoch_str, need_time, min(LRs), max(LRs), w_scheduler))

    # train for one epoch
    if args.procedure == "Simple-KD":
        kd_coef = 4. - 3.*(epoch / total_epoch)
        train_loss, train_acc1, train_acc5 = train_func(train_loader, teacher, network, criterion, w_scheduler, w_optimizer, optim_config, epoch_str, args.print_freq, logger, kd_coef)
    elif args.procedure == "basic":
        train_loss, train_acc1, train_acc5 = train_func(train_loader, network, criterion, w_scheduler, w_optimizer, optim_config, epoch_str, args.print_freq, logger)
    else:
        raise ValueError("invalid procedue name: {:}".format(args.procedure))
    train_time.update(time.time() - start_time)
    # log the results
    train_losses.append(train_loss); train_acc1s.append(train_acc1); train_acc5s.append(train_acc5)
    logger.log('***{:s}*** TRAIN [{:}] loss = {:.6f}, accuracy-1 = {:.2f}, accuracy-5 = {:.2f} time-cost = {:.1f}'.format(time_string(), epoch_str, train_loss, train_acc1, train_acc5, train_time.sum))

    # evaluate the performance
    if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
      logger.log('-'*150)
      if args.procedure == "Simple-KD":
          valid_loss, valid_acc1, valid_acc5 = valid_func(valid_loader, teacher, network, criterion, optim_config, epoch_str, args.print_freq_eval, logger)
      elif args.procedure == "basic":
          valid_loss, valid_acc1, valid_acc5 = valid_func(valid_loader, network, criterion, optim_config, epoch_str, args.print_freq_eval, logger)
      else:
          raise ValueError("invalid procedue name: {:}".format(args.procedure))
      valid_acc1s[epoch] = valid_acc1
      valid_losses.append(valid_loss); valid_acc5s.append(valid_acc5)
      logger.log('***{:s}*** VALID [{:}] loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f} | Best-Valid-Acc@1={:.2f}, Error@1={:.2f}'.format(time_string(), epoch_str, valid_loss, valid_acc1, valid_acc5, valid_acc1s['best'], 100-valid_acc1s['best']))
      if valid_acc1 > valid_acc1s['best']:
        valid_acc1s['best'] = valid_acc1
        find_best                = True
        logger.log('Currently, the best validation accuracy found at {:03d}-epoch :: acc@1={:.2f}, acc@5={:.2f}, error@1={:.2f}, error@5={:.2f}, save into {:}.'.format(epoch, valid_acc1, valid_acc5, 100-valid_acc1, 100-valid_acc5, model_best_path))
      num_bytes = torch.cuda.max_memory_cached( next(network.parameters()).device ) * 1.0
      logger.log('[GPU-Memory-Usage on {:} is {:} bytes, {:.2f} KB, {:.2f} MB, {:.2f} GB.]'.format(next(network.parameters()).device, int(num_bytes), num_bytes / 1e3, num_bytes / 1e6, num_bytes / 1e9))
      max_bytes[epoch] = num_bytes
    if epoch % 10 == 0: torch.cuda.empty_cache()

    # save checkpoint
    hint_results = (train_hint_losses, valid_hint_losses)
    train_results = (train_losses, train_acc1s, train_acc5s)
    valid_results = (valid_losses, valid_acc1s, valid_acc5s)
    save_path = save_checkpoint({
          'epoch'        : epoch,
          'args'         : deepcopy(args),
          'max_bytes'    : deepcopy(max_bytes),
          'FLOP'         : flop,
          'PARAM'        : param,
          'model_config' : student_config._asdict(),
          'genotype'     : genotype.tostr() if args.student_config is None else "",
          'nor_train_results'   : deepcopy(nor_train_results) if args.student_config is None and args.search_space_name == "nas-bench-201" else {},
          'optim_config' : optim_config._asdict(),
          'matching_optim_config' : matching_optim_config._asdict(),
          'student_model'   : student_model.state_dict(),
          'matching_layers' : matching_layers.state_dict(),
          'w_scheduler'    : w_scheduler.state_dict(),
          'w_optimizer'    : w_optimizer.state_dict(),
          'h_scheduler'    : h_scheduler.state_dict(),
          'h_optimizer'    : h_optimizer.state_dict(),
          'hint_results'    : deepcopy(hint_results),
          'train_results'   : deepcopy(train_results),
          'valid_results'   : deepcopy(valid_results),
          }, model_base_path, logger)
    if find_best: copy_checkpoint(model_base_path, model_best_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.log('\n' + '-'*200)
  logger.log('||| Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G'.format(param, flop, flop/1e3))
  logger.log('Finish training/validation in {:} and save final checkpoint into {:}'.format(convert_secs2time(epoch_time.sum, True), logger.path('info')))
  logger.log('-'*200 + '\n')
  logger.close()
  if args.student_config is None and args.search_space_name == "nas-bench-201":
      return arch_train_result['accuracy'], arch_valid_result['accuracy'], train_losses[-1], train_acc1s[-1], train_acc5s[-1], valid_losses[-1], valid_acc1s['best'], valid_acc5s[-1]
  else:
      return train_losses[-1], train_acc1s[-1], train_acc5s[-1], valid_losses[-1], valid_acc1s['best'], valid_acc5s[-1]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("fitnet-main")
  parser.add_argument('--exp_name',          type=str,  default="",   help='Experiment name')
  parser.add_argument('--overwrite',     type=bool,     default=False,  help='Overwrite the existing results')
  parser.add_argument('--resume'      ,     type=str,                   help='Resume path.')
  parser.add_argument('--init_model'  ,     type=str,                   help='The initialization model path.')
  # Save and Load
  parser.add_argument('--save_dir',           type=str,   default="./output/transfer-train",     help='Folder to save checkpoints and log.')
  parser.add_argument('--dataset',            type=str,   default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--data_path',          type=str,   default=os.environ['TORCH_HOME'] + "/cifar.python", help='Path to dataset')
  parser.add_argument('--cutout_length',    type=int,     default=-1,      help='The cutout length, negative means not use.')
  # channels and number-of-cells
  parser.add_argument('--max_nodes',          type=int,   default=4, help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   default=16, help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   default=5, help='The number of cells in one stage.')

  parser.add_argument('--beta', type=float, default=2e-2, help='matching loss scale')
  # NAS settings
  parser.add_argument('--search_space_name',  type=str,   default="nas-bench-201", help='The search space name.')
  parser.add_argument('--sample_method',    type=str,     default='uniform', choices=['uniform', 'infer', 'best'],    help='The architecture sampling method: uniform, infer, best')
  parser.add_argument('--select_num',         type=int,   default=100, help='The number of selected architectures to evaluate.')
  parser.add_argument('--arch_str',         type=str,       default=None, help="specific architecture to be transfer trained")
  # architecture leraning rate
  # parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  # parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  # parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  # Training Options
  parser.add_argument('--procedure'   ,     type=str,        default='basic',       help='The procedure basic prefix.')
  parser.add_argument('--KD_alpha'    ,     type=float,      default=0.9,           help='The alpha parameter in knowledge distillation.')
  parser.add_argument('--KD_temperature',   type=float,      default=4,           help='The temperature parameter in knowledge distillation.')
  # Printing
  parser.add_argument('--eval_frequency',   type=int,   default=1,      help='evaluation frequency (default: 200)')
  parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 100)')
  parser.add_argument('--print_freq_eval',  type=int,   default=100,    help='print frequency (default: 100)')
  parser.add_argument('--track_running_stats',type=int,   default=1, choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  # Configs and Checkpoints
  parser.add_argument('--student_config',     type=str,    default=None,      help='The path to the student model configuration')
  parser.add_argument('--optim_config',     type=str,    default="./configs/opts/CIFAR-fitnet-nas102.config",      help='The path to the optimizer configuration')
  parser.add_argument('--matching_optim_config',     type=str,  default="./configs/opts/CIFAR-fitnet-nas102-hint.config",        help='The path to the hint optimizer configuration')
  parser.add_argument('--student_checkpoint',    type=str,    default="./output/search-cell-nas-bench-102/SETN-cifar10/checkpoint/seed-36878-basic.pth",          help='The student checkpoint in knowledge distillation.')
  parser.add_argument('--teacher_checkpoint',    type=str,    default="./.latent-data/basemodels/cifar10/ResNet56.pth",          help='The teacher checkpoint in knowledge distillation.')
  # Acceleration
  parser.add_argument('--workers',          type=int,   default=4,      help='number of data loading workers (default: 8)')
  parser.add_argument('--batch_size',       type=int,   default=256,    help='batch size (default: 256)')
  # Random Seed
  parser.add_argument('--rand_seed',        type=int,   default=-1,     help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  if args.exp_name != "":
      if args.arch_str is None:
          args.save_dir = args.save_dir + "/" + args.exp_name + "/" + args.sample_method + "/" + args.procedure
      else:
          args.save_dir = args.save_dir + "/" + args.exp_name +  "/" + args.procedure

  results = main(args)
  if args.exp_name != "":
      try:
          write_results(args, results)
      except:
          print(results)
