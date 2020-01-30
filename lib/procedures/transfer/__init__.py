import time, torch
import torch.nn.functional as F
from log_utils import AverageMeter
from utils     import obtain_accuracy

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


def get_search_methods(nas_name, version=0):
    from .gdas_search  import search_func_gdas, search_func_gdas_v1, search_func_gdas_v2, search_func_gdas_v3, search_func_gdas_v4
    from .setn_search  import search_func_setn, search_func_setn_v2, search_func_setn_v3, search_func_setn_v4
    from .setn_search2 import search_w_setn2, search_a_setn2, search_w_setn2_v2, search_a_setn2_v2
    search_funcs = {'GDAS' : {0: search_func_gdas,
                              1: search_func_gdas_v1,
                              2: search_func_gdas_v2,
                              3: search_func_gdas_v3,
                              4: search_func_gdas_v4},
                    'SETN' : {0: search_func_setn,
                              1: search_func_setn,
                              2: search_func_setn_v2,
                              3: search_func_setn_v3,
                              4: search_func_setn_v4,
                              20: (search_w_setn2, search_a_setn2),
                              21: (search_w_setn2, search_a_setn2),
                              22: (search_w_setn2_v2, search_a_setn2_v2)}}
    return search_funcs[nas_name][version], valid_func
