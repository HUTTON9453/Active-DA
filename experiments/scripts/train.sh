#!/bin/bash

export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# != 7 ]
then
	echo "Please specify 1) cfg; 2) gpus; 3) method; 4) dataset; 5) strategy; 6) exp_name; 7) seed."
  exit
fi

cfg=${1}
gpus=${2}
method=${3}
dataset=${4}
strategy=${5}
exp_name=${6}
seed=${7}


out_dir=./img/decision_boundary/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}
out_dir=./img/confusion_matrix/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}
out_dir=./img/tsne/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}
out_dir=./img/count_bar/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}
out_dir=./img/scatter/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}
out_dir=./img/reliable_diagram/${exp_name}
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


#CUDA_VISIBLE_DEVICES=${gpus} /home/hutton/.pyenv/versions/2020_ADL/bin/python3 ./tools/train.py --cfg ${cfg} \
           #--method ${3} --exp_name ${4} --seed ${seed} 2>&1 | tee ${out_dir}/log.txt
            #--weights /tmp2/hutton/domain_adaptation_learning/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation/experiments/ckpt/office31_w2a_6_freezefc_MARGIN_SAMPLING/ckpt_finetune.weights \
CUDA_VISIBLE_DEVICES=${gpus} python3 ./tools/train.py \
            --seed ${seed}\
            --cfg ${cfg} \
           --method ${method} \
            --dataset ${dataset} \
            --strategy ${strategy} \
           --exp_name ${exp_name} --fix_fc True  2>&1 | tee ${out_dir}/log.txt
