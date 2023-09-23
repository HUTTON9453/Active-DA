# Improving Clustering Uncertainty-weighted Embeddings for Active Domain Adaptation
## Overview
Active-DA is a Python package designed to make active domain adaptation easier for researchers and real-world users. Active-DA based on [DeepAL](https://github.com/ej0cl6/deep-active-learning). In our work, we analyze the state-of-the-art of active domain adaptation, [ADA-CLUE](https://arxiv.org/pdf/2010.08666.pdf). We observed that uncertainty is not reliable in early stage of training. Thus in this package, we proposed a intuition method, Loop threshold, to improve the performace of CLUE. We introduce a threshold to control when to using uncertainty-weighted clustering or constant-weighted clustering. We called Clustering Uncertainty-weighted Embeddings with Loop Threshold as simple solution. And in this package we not only implement popular active learning methods for domain adaptation but also provides benchmark results on cross-domain dataset. 

## Requirements
- Python 3.7
- Pytorch 1.1
- PyYAML 5.1.1

## Strategy
We provide some baseline strategies as well as some state-of-the-are strategies in this package as the following:

* Random Sampling
* Margin Sampling
* Entropy Sampling
* [Coreset](https://arxiv.org/abs/1708.00489)
* [KMeans Sampling](https://arxiv.org/abs/1708.00489)
* [BADGE](http://arxiv.org/abs/1906.03671)
* [AADA](https://openaccess.thecvf.com/content_WACV_2020/papers/Su_Active_Adversarial_Domain_Adaptation_WACV_2020_paper.pdf)
* [CLUE](https://arxiv.org/pdf/2010.08666.pdf)
* [Density Weighted Uncertainty Sampling (the variant of CLUE)(Link coming soon)]()
* [CLUE with Loop threshold(Link coming soon)]()
## Dataset
The structure of the dataset should be like

```
Office-31
|_ category.txt
|_ amazon
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ dslr
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ ...
```
The "category.txt" contains the names of all the categories, which is like
```
back_pack
bike
bike_helmet
...
```


## Training
```
./experiments/scripts/train.sh ${config_yaml} ${gpu_ids} ${domain_adaptation_solver} ${dataset} ${active_learning_method} ${experiment_name} ${seed}
```
For example, for the Office-31 dataset,
```
./experiments/scripts/train.sh ./experiments/config/Office-31/MME/office31_train_amazon2dslr_cfg.yaml 0 CAN office31 OurSampling office31_a2d 1126
```

The experiment log file and the saved checkpoints will be stored at ./experiments/ckpt/${experiment_name}


## Contact
If you have any questions, please contact me via r08922134@csie.ntu.edu.tw.

## Acknowledgement
The authors thank members of the Computational Learning Lab at National Taiwan University.

