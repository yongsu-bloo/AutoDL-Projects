##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from .starts     import prepare_seed, prepare_logger, get_machine_info, save_checkpoint, copy_checkpoint
from .optimizers import get_optim_scheduler

def get_procedures(procedure):
  from .basic_main     import basic_train, basic_valid
  from .search_main    import search_train, search_valid
  from .search_main_v2 import search_train_v2
  from .simple_KD_main import simple_KD_train, simple_KD_valid
  from .fitnet_main import fitnet_train, fitnet_valid

  train_funcs = {'basic' : basic_train, \
                 'search': search_train,'Simple-KD': simple_KD_train, \
                 'search-v2': search_train_v2, \
                 'fitnet': fitnet_train}
  valid_funcs = {'basic' : basic_valid, \
                 'search': search_valid,'Simple-KD': simple_KD_valid, \
                 'search-v2': search_valid,    \
                 'fitnet': fitnet_valid}

  train_func  = train_funcs[procedure]
  valid_func  = valid_funcs[procedure]
  return train_func, valid_func

def get_search_methods(nas_name, version=0):
    from .transfer_search_main import search_func_gdas, search_func_gdas_v1, search_func_gdas_v2, search_func_gdas_v3, search_func_gdas_v4, \
                                      search_func_setn, search_func_setn_v5, search_func_setn_v6, valid_func
    search_funcs = {'GDAS' : {0: search_func_gdas,
                              1: search_func_gdas_v1,
                              2: search_func_gdas_v2,
                              3: search_func_gdas_v3,
                              4: search_func_gdas_v4},
                    'SETN' : {0: search_func_setn,
                              1: search_func_setn,
                              5: search_func_setn_v5,
                              6: search_func_setn_v6}}
    return search_funcs[nas_name][version], valid_func
