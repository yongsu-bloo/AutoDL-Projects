#!/bin/bash
# bash ./MetaENAS.sh cifar10 0 -1 exp_name
echo script name: $0
echo $# arguments
if [ "$#" -lt 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, BN-tracking-status, and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
BN=$2
seed=$3
exp_name=$4
channel=12
num_cells=2
max_nodes=4
space=nas-bench-201

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth
benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth

save_dir=./output/search-cell-${space}/MetaENAS-${dataset}-BN${BN}

OMP_NUM_THREADS=4 python ./exps/algos/MetaENAS.py \
  --exp_name ${exp_name} \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--track_running_stats ${BN} \
	--config_path ./configs/research/MetaENAS500.config \
  --n_shot 1 \
	--controller_entropy_weight 0.0001 \
	--controller_bl_dec 0.99 \
	--controller_train_steps 5 \
	--controller_num_aggregate 10 \
	--controller_num_samples 100 \
	--workers 6 --print_freq 200 --rand_seed ${seed}
  # --load_supernet /home/server19/yongsu/AutoDL-Projects/output/search-cell-nas-bench-201/MetaENAS-cifar10-BN0/meta-1shot-32task/seed-52983-supernet.pth
  # --load_supernet /home/server19/yongsu/AutoDL-Projects/output/search-cell-nas-bench-201/MetaENAS-cifar10-BN0/nometa-1shot/seed-8476-supernet.pth
