# +
import torch
import torch.nn as nn
import os
from utils.utils import to_cuda, to_onehot, draw, visualize_2d, visualize_3d
from . import utils as solver_utils
import data.utils as data_utils
from torch import optim
from data.custom_dataset_dataloader import CustomDatasetDataLoader2
from data import single_dataset
from math import ceil, exp
from .base_solver import BaseSolver
from tensorboardX import SummaryWriter
from model.model import Discriminator
from torch import optim
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import grad
from data.image_folder import make_dataset_with_labels
from sklearn.manifold import TSNE


class AADASolver(BaseSolver):
    def __init__(self, net, dataloader, X, Y, bn_domain_map={}, resume=None, \
                  idxs_lb=None, classnames=None, **kwargs):
        super(AADASolver, self).__init__(net, dataloader, X, Y, \
                      bn_domain_map=bn_domain_map, resume=resume, \
                      idxs_lb=idxs_lb, classnames=classnames, **kwargs)
        assert(len(self.train_data) > 0), "Please specify the training domain."
        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}
        self.accu = 0
        self.alpha = 0.1
        self.source_len = len(self.train_data[self.source_name]['loader'].dataset)
        self.target_len = len(self.train_data[self.target_name]['loader'].dataset)
        self.writer = SummaryWriter(f'./runs/{self.opt.EXP_NAME}')
        self.discriminator = to_cuda(nn.DataParallel(Discriminator()))
        # self.discriminator.module.set_bn_domain()
        net_param_groups = solver_utils.set_param_groups(self.net, dict({}))
        discriminator_param_groups = solver_utils.set_param_groups(self.discriminator, dict({}))
        # self.discriminator_optimizer = optim.Adam(param_groups, lr=0.001)
        self.optimizer = optim.SGD(net_param_groups+discriminator_param_groups, lr=0.005, momentum=self.opt.TRAIN.MOMENTUM, weight_decay=self.opt.TRAIN.WEIGHT_DECAY)
        self.discriminator_criterion = nn.CrossEntropyLoss().cuda()
        # self.discriminator_optimizer = optim.RMSprop(param_groups, lr=0.001)
        if self.opt.FINETUNE.NUM_QUERY_PERCENT != None:
            self.opt.FINETUNE.NUM_QUERY = math.floor(len(self.train_data[self.target_name]['loader'].dataset) * self.opt.FINETUNE.NUM_QUERY_PERCENT / self.opt.FINETUNE.NUM_ROUND)

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb


    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        score = self.score(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        score_sorted, idxs = score.sort(descending=True)
        idxs = idxs_unlabeled[idxs[:n].cpu()]

        return idxs

    def random_sampling(self, n):
        idxs = np.random.choice(np.where(self.idxs_lb==0)[0], n)
        return idxs


    def score(self, X, Y):
        train_transform = data_utils.get_transform(False)
        dataset_type = 'SingleDataset'
        dataset = getattr(single_dataset, dataset_type)(root="")
        dataset.initialize(root="",
                                transform=train_transform, classnames=self.classnames,
                                data_paths=X, data_labels=Y)
        loader_te = DataLoader(dataset,
                            shuffle=False, batch_size=self.opt.FINETUNE.TEST_BATCH_SIZE, num_workers=self.opt.FINETUNE.TEST_NUM_WORKERS)
        
        self.net.eval()
        self.discriminator.eval()
        s = torch.zeros([len(Y)])
        with torch.no_grad():
            for data in loader_te:
                x, y = to_cuda(data['Img']), to_cuda(data['Label'])
                out = self.net(x)
                probs = out['probs']
                df = self.discriminator(out['feat'], self.alpha)
                df = df[:,1] / df.sum(dim=1)
                w = (1-df) / df
                temp_s = w * solver_utils.H(probs)
                s[data['Index']] = temp_s.cpu()

        return s

    def build_active_learning_dataloader(self):
        train_transform = data_utils.get_transform(True)
        if self.loop == 0:
            q_idxs = self.random_sampling(self.opt.FINETUNE.NUM_QUERY)
        else:
            q_idxs = self.query(self.opt.FINETUNE.NUM_QUERY)
        '''
        dataset_type = 'SingleDataset'
        target_train_dataset = getattr(single_dataset, dataset_type)()
        target_train_dataset.initialize_data(data_paths=self.X[np.sort(q_idxs)], data_labels=self.Y[np.sort(q_idxs)], transform=self.train_transform)
        dataloaders = CustomDatasetDataLoader2(dataset=target_train_dataset, batch_size=100,train=False, num_workers=1,classnames=self.train_data[self.target_name]['loader'].classnames)
        draw(self.net, self.bn_domain_map,dataloaders , self.opt, "test_r"+str(self.loop))
        '''
        self.idxs_lb[q_idxs] = True
        '''
        dataset_type = 'SingleDataset'
        target_train_dataset = getattr(single_dataset, dataset_type)()
        target_train_dataset.initialize_data(data_paths=self.X[self.idxs_lb], data_labels=self.Y[self.idxs_lb], transform=self.train_transform)
        dataloaders = CustomDatasetDataLoader2(dataset=target_train_dataset, batch_size=100,train=False, num_workers=1,classnames=self.train_data[self.target_name]['loader'].classnames)
        draw(self.net, self.bn_domain_map,dataloaders, self.opt, "all_target_label_data_r"+str(self.loop))
        
        dataset_type = 'SingleDataset'
        target_train_dataset = getattr(single_dataset, dataset_type)()
        target_train_dataset.initialize_data(data_paths=self.X, data_labels=self.Y, transform=self.train_transform)
        dataloaders = CustomDatasetDataLoader2(dataset=target_train_dataset, batch_size=100,train=False, num_workers=1,classnames=self.train_data[self.target_name]['loader'].classnames)
        draw(self.net, self.bn_domain_map,dataloaders , self.opt, "all_target_data_r"+str(self.loop))
        
        self.visualize(dim=2, plot_num=2000)
        '''
        print("Num of query samples", sum(self.idxs_lb))
        train_transform = data_utils.get_transform(True)
        # rebuild source
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        batch_size = self.opt.TRAIN.SOURCE_BATCH_SIZE
        dataset_type = 'SingleDataset'
        print('Building %s dataloader...' % self.source_name)
        self.train_data[self.source_name]['loader'] = CustomDatasetDataLoader(dataset_root="", dataset_type=dataset_type, 
                                                      batch_size=batch_size, transform=train_transform, 
                                                      data_paths=np.concatenate((self.X[idxs_train], np.array(self.train_data[self.source_name]['loader'].dataset.data_paths[:self.source_len]))), data_labels=np.concatenate((self.Y[idxs_train], np.array(self.train_data[self.source_name]['loader'].dataset.data_labels[:self.source_len]))),
                                                      train=True, num_workers=self.opt.NUM_WORKERS, 
                                                      classnames=self.classnames)
        '''
        source_train_dataset = getattr(single_dataset, dataset_type)()
        source_train_dataset.initialize_data(data_paths=np.concatenate((self.X[idxs_train], np.array(self.train_data[self.source_name]['loader'].dataset.data_paths[:self.source_len]))), data_labels=np.concatenate((self.Y[idxs_train], np.array(self.train_data[self.source_name]['loader'].dataset.data_labels[:self.source_len]))), transform=train_transform)
        self.train_data[self.source_name]['loader'] = CustomDatasetDataLoader2(
                    dataset=source_train_dataset, batch_size=batch_size,
                    train=True, num_workers=self.opt.NUM_WORKERS,
                    classnames=self.train_data[self.source_name]['loader'].classnames)
        '''
        # rebuild target
        idxs_train = np.arange(self.n_pool)[~self.idxs_lb]
        batch_size = self.opt.TRAIN.TARGET_BATCH_SIZE
        dataset_type = 'SingleDataset'
        print('Building %s labeled dataloader...' % self.target_name)
        self.train_data[self.target_name]['loader'] = CustomDatasetDataLoader(dataset_root="", dataset_type=dataset_type, 
                                                      batch_size=batch_size, transform=train_transform, 
                                                      data_paths=self.X[idxs_train], data_labels=self.Y[idxs_train],
                                                      train=True, num_workers=self.opt.NUM_WORKERS, 
                                                      classnames=self.classnames)
        '''
        dataset_type = 'SingleDataset'
        target_train_dataset = getattr(single_dataset, dataset_type)()
        target_train_dataset.initialize_data(data_paths=self.X[idxs_train], data_labels=self.Y[idxs_train], transform=train_transform)
        self.train_data[self.target_name]['loader'] = CustomDatasetDataLoader2(
                dataset=target_train_dataset, batch_size=self.opt.TRAIN.TARGET_BATCH_SIZE,
                train=True, num_workers=self.opt.NUM_WORKERS,
                classnames=self.train_data[self.target_name]['loader'].classnames)
        '''


    def solve(self):
        if self.resume:
            self.iters += 1
            self.loop += 1

        self.compute_iters_per_loop()
        
        for i in range(self.opt.AADA.MAX_EPOCH):
            self.update_network()
            self.loop += 1

        for j in range(self.opt.FINETUNE.NUM_ROUND):
            self.build_active_learning_dataloader()
            for i in range(self.opt.AADA.MAX_EPOCH):
                self.update_network()
                self.loop += 1

        print('Training Done!')

    def compute_iters_per_loop(self):
        self.iters_per_loop = max(len(self.train_data[self.source_name]['loader']), len(self.train_data[self.target_name]['loader']))
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self):
        # initial configuration
        stop = False
        update_iters = 0
        self.train_data[self.source_name]['iterator'] = iter(self.train_data[self.source_name]['loader'])
        self.train_data[self.target_name]['iterator'] = iter(self.train_data[self.target_name]['loader'])
        while not stop:
            loss = 0
            # update learning rate
            #self.update_lr()

            # set the status of network
            self.net.train()
            self.discriminator.train()
            self.net.zero_grad()
            self.discriminator.zero_grad()

			# get sample from source and target
            source_sample = self.get_samples(self.source_name)
            source_data, source_gt = source_sample['Img'],\
                          source_sample['Label']
            target_sample = self.get_samples(self.target_name)
            target_data = target_sample['Img']

            target_label_index = np.isin(target_sample['Path'], self.X[self.idxs_lb])
            target_label_data, target_label_gt = target_sample['Img'][target_label_index], \
                                target_sample['Label'][target_label_index]

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            target_data = to_cuda(target_data)
            target_label_data = to_cuda(target_label_data)
            target_label_gt = to_cuda(target_label_gt)

            # compute cross entropy loss
            source_preds = self.net(source_data)['logits']
            ce_loss = self.CELoss(source_preds, source_gt)

            # compute domain loss
            p = float(self.iters_per_loop * (self.loop % self.opt.AADA.MAX_EPOCH) +  update_iters) / (self.opt.AADA.MAX_EPOCH * self.iters_per_loop)
            self.alpha = 2. / (1. + np.exp(-10 * p)) - 1

            combined_image = torch.cat((source_data, target_data), 0)
            combined_feature = self.net(combined_image)['feat']
            domain_pred = self.discriminator(combined_feature, self.alpha)

            domain_source_labels = torch.zeros(source_data.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_data.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_loss = self.discriminator_criterion(domain_pred, domain_combined_label)

            # if np.sum(target_label_index) > 0:
                # target_preds = self.net(target_label_data)['logits']
                # target_ce_loss = self.CELoss(target_preds, target_label_gt)
                # # entropy_loss_target = H(class_output_target).sum()
            # else:
                # target_ce_loss = 0
                # # entropy_loss_target = 0


            loss = self.opt.AADA.LAMBDA*domain_loss + ce_loss
            loss.backward()

            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = self.model_eval(source_preds, source_gt)
                cur_loss = {"alpha": self.alpha, 'ce_loss': ce_loss,'domain_loss': domain_loss, 'loss':loss}
                self.writer.add_scalar("ce_loss", ce_loss, self.iters)
                self.writer.add_scalar("domain_loss", domain_loss, self.iters)
                self.writer.add_scalar("total_loss", loss, self.iters)
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

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.train_transform),
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
