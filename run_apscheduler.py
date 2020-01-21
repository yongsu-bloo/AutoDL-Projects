"""
This code is mainly from https://github.com/sehkmg/apscheduler_for_machine_learning
"""
from datetime import datetime, timedelta
import time
import os, sys
import json

from apscheduler.schedulers.background import BackgroundScheduler
from threading import Lock

from copy import deepcopy
from math import pow
import numpy as np
import pandas as pd
np.random.seed(777)

def main(config_path, base_command):
    # parse json file
    with open(config_path) as json_file:
        settings = json.load(json_file)

    # random search trial repetition
    trial_num = settings['trial_num']

    # base command
    base_command = [base_command]

    # gpu ids
    gpu_ids = settings['gpu_ids']
    exp_per_gpu = trial_num // len(gpu_ids)
    parallel_tasks_per_gpu = settings['parallel_tasks_per_gpu']

    # gpu command
    gpu_ids = [['CUDA_VISIBLE_DEVICES={:d}'.format(i)] for i in gpu_ids]

    # arguments
    args_info = settings['args_info']
    exp_name = settings['exp_name']
    with open('./results/result-{}.csv'.format(exp_name), 'w') as res:
        keys = [arg['arg'][2:] for arg in args_info]
        metrics = ["nor_train_acc1",
                   "nor_valid_acc1",
                   'train_loss',
                   'train_acc1',
                   'train_acc5',
                   'valid_loss',
                   'valid_acc1',
                   'valid_acc5']
        title = keys + metrics
        title = ','.join(title)
        res.write(title + '\n')

    # prepare argument info
    arg_list = [[]]

    for arg_info in args_info:
        cur_len = len(arg_list)
        for i in range(cur_len):
            base_arg = arg_list.pop(0)

            arg = deepcopy(base_arg)

            for option in arg_info['option']:
                arg.append('{}={}'.format(arg_info['arg'], option))
                arg_list.append(arg)
                arg = deepcopy(base_arg)

    # permute arg list for random search
    arg_list = np.random.permutation(arg_list).tolist()

    command_lists = [[] for i in range(len(gpu_ids))]

    # combine arguments
    with open('./results/command_list-{}.txt'.format(exp_name), 'w') as cmd_file:
        for i, gpu in enumerate(gpu_ids):
            for j in range(exp_per_gpu):
                if j == (exp_per_gpu // parallel_tasks_per_gpu):
                    cmd_file.write('\n')

                cmd = ' '.join(gpu_ids[i] + base_command + arg_list[j * len(gpu_ids) + i])
                cmd_file.write(cmd)
                cmd_file.write('\n')

                command_lists[i].append(cmd)

            if len(gpu_ids) * exp_per_gpu + i < trial_num:
                cmd = ' '.join(gpu_ids[i] + base_command + arg_list[exp_per_gpu * len(gpu_ids) + i])
                cmd_file.write(cmd)
                cmd_file.write('\n')

                command_lists[i].append(cmd)

            cmd_file.write('\n')

    # divide tasks
    def chunk(seq, size):
        return (seq[i::size] for i in range(size))
    # chunk makes list of list
    command_lists = [list(chunk(cmd,parallel_tasks_per_gpu))for cmd in command_lists]
    # flatten the list of list to a single list
    command_lists = [cmd for cmd_list in command_lists for cmd in cmd_list]

    # store total commands to know when to stop scheduler
    total = 0

    # define counter and counter lock to check how many commands executed
    counter_lock = Lock()
    counter = [0]

    # job definition
    def execute(command_list, counter):
        for command in command_list:
            os.system(command)

            # add counter when each commands ends
            counter_lock.acquire()
            counter[0] += 1
            counter_lock.release()

    # init scheduler
    executors = {
        'default': {'type': 'threadpool', 'max_workers': 60}
    }

    scheduler = BackgroundScheduler(executors=executors)
    scheduler.start()

    # store total commands
    for command_list in command_lists:
        total += len(command_list)

    # deploy commands
    delta = timedelta(seconds=1)
    for i, command_list in enumerate(command_lists):
        scheduler.add_job(execute, 'date', run_date=datetime.now()+delta*(i+1), args=[command_list, counter])

    # scheduler handling part
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        # if every jobs ended, stop
        while counter[0] < total:
            time.sleep(10)
            res = pd.read_csv('./results/result-{}.csv'.format(exp_name), sep=',', header=0)
            res = res.sort_values('valid_acc1', ascending=False)
            res.to_csv('./results/result-{}.csv'.format(exp_name), sep=',', index=False)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()

if __name__=='__main__':
    config_path = sys.argv[1]
    base_command = sys.argv[2] if len(sys.argv) >= 3 else 'OMP_NUM_THREADS=4 python exps/fitnet-train.py'
    start_time = time.time()
    main(config_path, base_command)
    print("Total Time to Experiment: " + str(timedelta(seconds=time.time()-start_time)))
