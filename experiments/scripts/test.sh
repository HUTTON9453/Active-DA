#!/bin/bash

export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# != 4 ]
then
  echo "Please specify 1) cfg; 2) gpus; 3) exp_name; 4) seed."
  exit
fi

cfg=${1}
gpus=${2}
adapted=True
exp_name=${3}
seed=${4}

out_dir=./img/confusion_matrix/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}
out_dir=./img/decision_boundary/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}
out_dir=./experiments/ckpt/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}

CUDA_VISIBLE_DEVICES=${gpus} python3 ./tools/test.py --cfg ${cfg} --seed ${seed} --adapted \
       --exp_name ${exp_name} 2>&1 | tee ${out_dir}/log.txt
