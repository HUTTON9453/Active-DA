# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from . import utils as solver_utils
import data.utils as data_utils
from utils.utils import to_cuda, to_onehot, draw, draw_count_bar, visualize_2d, visualize_3d
from data import single_dataset
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from torch import optim
from . import clustering
from discrepancy.cdd import CDD
from math import ceil as ceil
from .base_solver import BaseSolver
from torch.utils.data import DataLoader
from copy import deepcopy
from tensorboardX import SummaryWriter
import math
from data.image_folder import make_dataset_with_labels
from sklearn.manifold import TSNE
import pickle
from collections import defaultdict
from datetime import datetime
from data.dataset import get_handler


class FTSolver(BaseSolver):
    def __init__(self, net, dataloader, X, Y, bn_domain_map={}, resume=None, \
                  idxs_lb=None, classnames=None, **kwargs):
        super(FTSolver, self).__init__(net, dataloader, X, Y, \
                      bn_domain_map=bn_domain_map, resume=resume, \
                    idxs_lb=idxs_lb, classnames=classnames, **kwargs)
        assert(len(self.train_data) > 0), "Please specify the training domain."
        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}
        self.accu = 0
        self.alpha = 0.1
        self.source_len = len(self.train_data[self.source_name]['loader'].dataset)
        self.target_len = len(self.train_data[self.target_name]['loader'].dataset)
        self.writer = SummaryWriter(f'./experiments/runs/FT/{self.opt.EXP_NAME}')
        if self.opt.FINETUNE.NUM_QUERY_PERCENT != None:
            self.opt.FINETUNE.NUM_QUERY = math.floor(len(self.train_data[self.target_name]['loader'].dataset) * self.opt.FINETUNE.NUM_QUERY_PERCENT / self.opt.FINETUNE.NUM_ROUND)
    

    def build_active_learning_dataloader(self, index):
        q_idxs = self.query(self.opt.FINETUNE.NUM_QUERY)
        train_transform = data_utils.get_transform(True)
        dataset_type = 'SingleDataset'
        '''
        dataloaders = CustomDatasetDataLoader(dataset_root="", dataset_type=dataset_type, 
                                                  batch_size=100, transform=train_transform, 
                                                  data_paths=self.X[np.sort(q_idxs)], data_labels=self.Y[np.sort(q_idxs)],
                                                  train=False, num_workers=1, 
                                                  classnames=self.classnames)
        draw(self.net, self.bn_domain_map,dataloaders , self.opt, "query_data"+str(self.loop))

        
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        dataloaders = CustomDatasetDataLoader(dataset_root="", dataset_type=dataset_type, 
                                                  batch_size=100, transform=train_transform, 
                                                  data_paths=self.X[self.idxs_lb], data_labels=self.Y[self.idxs_lb],
                                                  train=False, num_workers=1, 
                                                  classnames=self.classnames)
        draw(self.net, self.bn_domain_map,dataloaders, self.opt, "all_target_label_data_r"+str(self.loop))
        '''
        self.idxs_lb[q_idxs] = True
        '''
        dataloaders = CustomDatasetDataLoader(dataset_root="", dataset_type=dataset_type, 
                                                  batch_size=100, transform=train_transform, 
                                                  data_paths=self.X, data_labels=self.Y,
                                                  train=False, num_workers=1, 
                                                  classnames=self.classnames)
        
        draw(self.net, self.bn_domain_map,dataloaders, q_idxs, self.opt, "r"+str(self.loop))
        
        count = np.bincount(self.Y[q_idxs], minlength=self.opt.DATASET.NUM_CLASSES)
        draw_count_bar("./img/count_bar/"+self.opt.EXP_NAME, count, self.opt.DATASET.NUM_CLASSES, "R"+str(self.loop))
        self.visualize(dim=2, plot_num=2000)
        '''
        print("Num of query samples", sum(self.idxs_lb))

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        batch_size = min(len(idxs_train), self.opt.TRAIN.TARGET_BATCH_SIZE)
        dataset_type = 'SingleDataset'
        print('Building %s labeled dataloader...' % self.target_name)
        self.train_data[self.target_name+'_labeled']['loader'] = CustomDatasetDataLoader(dataset_root="", dataset_type=dataset_type, 
                                                      batch_size=batch_size, transform=train_transform, 
                                                      data_paths=self.X[idxs_train], data_labels=self.Y[idxs_train],
                                                      train=True, num_workers=self.opt.NUM_WORKERS, 
                                                      classnames=self.classnames)


    def solve(self):
        accu = defaultdict(list)
        if self.resume:
            self.iters += 1
            self.loop += 1

        with torch.no_grad():
            accu[0.0] = self.test()
        if self.opt.TRAIN.LR_INIT != 0.0:
            self.base_lr = self.opt.TRAIN.LR_INIT
            self.update_lr_one_round()
            
        for j in range(self.opt.FINETUNE.NUM_ROUND):
            self.build_active_learning_dataloader(j%5)
            self.compute_iters_per_loop()
            for i in range(self.opt.FT.MAX_EPOCH):
                self.update_network()
                self.loop += 1
            with torch.no_grad():
                accu[(j+1)*self.opt.FINETUNE.NUM_QUERY] = self.test()
        self.log(accu)

        print('Training Done!')

    def compute_iters_per_loop(self):
        self.iters_per_loop = len(self.train_data[self.target_name+'_labeled']['loader'])
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self):
        # initial configuration
        stop = False
        update_iters = 0
        if self.train_data[self.target_name+'_labeled']['loader'] is not None:
            self.train_data[self.target_name+'_labeled']['iterator'] = iter(self.train_data[self.target_name+'_labeled']['loader'])
        while not stop:
            loss = 0
            # update learning rate
            if self.opt.TRAIN.LR_SCHEDULE != '':
                self.update_lr()


            # set the status of network
            self.net.train()
            #self.net.zero_grad()
            self.optimizer.zero_grad()
			
            if self.train_data[self.target_name+'_labeled']['loader'] is not None:
                target_label_sample = self.get_samples(self.target_name+"_labeled")
                target_label_data, target_label_gt = target_label_sample['Img'],\
                              target_label_sample['Label']

            
            if self.train_data[self.target_name+'_labeled']['loader'] is not None:
                target_label_data = to_cuda(target_label_data)
                target_label_gt = to_cuda(target_label_gt)

            if self.train_data[self.target_name+'_labeled']['loader'] is not None:
                target_preds = self.net(target_label_data)['logits']
                target_ce_loss = self.CELoss(target_preds, target_label_gt)
            else:
                target_ce_loss = 0
            loss = target_ce_loss
            loss.backward()
            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = 0
                cur_loss = {'target_ce_loss': target_ce_loss}
                self.writer.add_scalar("target_ce_loss", target_ce_loss, self.iters)
                self.logging(cur_loss, accu)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop,
                              self.iters, self.opt.EVAL_METRIC, accu))
                self.writer.add_scalar("eval acc", accu, self.iters)

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False


    def visualize(self, dim=2, plot_num=1000):
        print("t-SNE reduces to dimension {}".format(dim))

        self.net.eval()
        handler = get_handler()
        train_transform = data_utils.get_transform(True)
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(handler(self.X[idxs_train], self.Y[idxs_train], transform=train_transform),
                            shuffle=True, batch_size=self.opt.FINETUNE.TRAIN_BATCH_SIZE, num_workers=self.opt.FINETUNE.TRAIN_NUM_WORKERS)

        total_src_z = None#to_cuda(torch.FloatTensor())
        total_tar_z = None#to_cuda(torch.FloatTensor())
        total_tar_label_z = None#to_cuda(torch.FloatTensor())

        ''' If use USPS dataset, change it to IntTensor() '''
        src_label = torch.IntTensor()
        tar_label = torch.IntTensor()
        total_tar_label = torch.IntTensor()

        for index, src in enumerate(self.train_data[self.source_name]['loader']):
            data, label = to_cuda(src['Img']), src['Label']
            #src_data = torch.cat((src_data, data))
            src_label = torch.cat((src_label, label))
            src_z = self.net(data)['feat']
            #total_src_z = torch.cat((total_src_z, src_z))
            if total_src_z is None:
                total_src_z = src_z.cpu().detach().numpy()
            else:
                total_src_z = np.concatenate((total_src_z, src_z.cpu().detach().numpy()))

        for index, tar in enumerate(self.train_data[self.target_name]['loader']):
            data, label = to_cuda(tar['Img']), tar['Label']
            #tar_data = torch.cat((tar_data, data))
            tar_label = torch.cat((tar_label, label))
            tar_z = self.net(data)['feat']
            if total_tar_z is None:
                total_tar_z = tar_z.cpu().detach().numpy()
            else:
                total_tar_z = np.concatenate((total_tar_z, tar_z.cpu().detach().numpy()))

                
        for index, (x, y, idxs) in enumerate(loader_tr):
            x = to_cuda(x)
            total_tar_label = torch.cat((total_tar_label, y))
            tar_z = self.net(x)['feat']
            if total_tar_label_z is None:
                total_tar_label_z = tar_z.cpu().detach().numpy()
            else:
                total_tar_label_z = np.concatenate((total_tar_label_z, tar_z.cpu().detach().numpy()))

        ''' for MNIST dataset '''
        #if src_data.shape[1] != 3:
        #    src_data = src_data.expand(
        #        src_data.shape[0], 3, self.img_size, self.img_size)

        #if plot_num > len(src_data):
        #    src_plot_num = len(src_data)
        #else:
        #    src_plot_num = plot_num
        #if plot_num > len(tar_data):
        #    tar_plot_num = len(tar_data)
        #else:
        #    tar_plot_num = plot_num
        #src_data, src_label = src_data[0:src_plot_num], src_label[0:src_plot_num]
        #tar_data, tar_label = tar_data[0:tar_plot_num], tar_label[0:tar_plot_num]

        #src_z = self.net(src_data)['feat']
        #tar_z = self.net(tar_data)['feat']

        data = np.concatenate((total_tar_label_z, total_src_z, total_tar_z))
        label = np.concatenate((src_label.numpy(), tar_label.numpy()))
        src_tag = torch.zeros(total_src_z.shape[0])
        tar_tag = torch.ones(total_tar_z.shape[0])
        tag = np.concatenate((src_tag.numpy(), tar_tag.numpy()))

        ''' t-SNE process '''
        tsne = TSNE(n_components=dim)

        embedding = tsne.fit_transform(data)

        embedding_max, embedding_min = np.max(
            embedding, 0), np.min(embedding, 0)
        embedding = (embedding-embedding_min) / (embedding_max - embedding_min)

        if dim == 2:
            visualize_2d("./img/tsne/"+self.opt.EXP_NAME, embedding[total_tar_label_z.shape[0]:],
                         label, tag, embedding[0:total_tar_label_z.shape[0]], total_tar_label.numpy(), self.opt.DATASET.NUM_CLASSES, str(self.loop))

        elif dim == 3:
            visualize_3d("./img/tsne/"+self.opt.EXP_NAME, embedding[total_tar_label_z.shape[0]:],
                         label, tag, embedding[0:total_tar_label_z.shape[0]], total_tar_label.numpy(), self.opt.DATASET.NUM_CLASSES, str(self.loop))
