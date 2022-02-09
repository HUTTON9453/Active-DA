import torch
import torch.nn as nn
import os
from . import utils as solver_utils
import data.utils as data_utils
from utils.utils import to_cuda, mean_accuracy, accuracy, visualize_2d
from torch import optim
from math import ceil as ceil, floor as floor
from config.config import cfg
from data import single_dataset
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
from .kCenterGreedy import kCenterGreedy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import random
import json
from utils.utils import make_model_diagrams
from sklearn.metrics import brier_score_loss

class BaseSolver:
    def __init__(self, net, dataloader, X, Y, bn_domain_map={}, resume=None, idxs_lb=None, classnames=None, **kwargs):
        self.opt = cfg
        self.idxs_lb = idxs_lb
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n_pool = len(Y)
        self.source_name = self.opt.DATASET.SOURCE_NAME
        self.target_name = self.opt.DATASET.TARGET_NAME
        self.classnames = classnames

        self.net = net
        self.init_data(dataloader)
        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda()

        self.loop = 0
        self.iters = 0
        self.iters_per_loop = None
        self.history = {}

        self.base_lr = self.opt.TRAIN.BASE_LR
        self.momentum = self.opt.TRAIN.MOMENTUM

        self.bn_domain_map = bn_domain_map

        self.optim_state_dict = None
        self.resume = False
        if resume is not None:
            self.resume = True
            self.loop = resume['loop']
            self.iters = resume['iters']
            self.history = resume['history']
            self.optim_state_dict = resume['optimizer_state_dict']
            self.bn_domain_map = resume['bn_domain_map']
            print('Resume Training from loop %d, iters %d.' % \
			(self.loop, self.iters))

        self.build_optimizer()

    def init_data(self, dataloader):
        self.train_data = {key: dict() for key in dataloader if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloader:
                continue
            cur_dataloader = dataloader[key]
            self.train_data[key]['loader'] = cur_dataloader
            self.train_data[key]['iterator'] = None

        if 'test' in dataloader:
            self.test_data = dict()
            self.test_data['loader'] = dataloader['test']

    def build_optimizer(self):
        opt = self.opt
        # for name, param in self.net.named_parameters():
            # if 'FC' in name:
                # param.requires_grad = False
        self.active_learning_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.opt.FINETUNE.LR, momentum=self.opt.FINETUNE.MOMENTUM)
        param_groups = solver_utils.set_param_groups(self.net,
		dict({'FC': opt.TRAIN.LR_MULT}))
        # for name, param in self.net.named_parameters():
            # if 'FC' in name:
                # param.requires_grad = True

        assert opt.TRAIN.OPTIMIZER in ["Adam", "SGD"], \
            "Currently do not support your specified optimizer."

        if opt.TRAIN.OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(param_groups,
			lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2],
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

        elif opt.TRAIN.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(param_groups,
			lr=self.base_lr, momentum=self.momentum,
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

        if self.optim_state_dict is not None:
            self.optimizer.load_state_dict(self.optim_state_dict)

    def update_lr_one_round(self):
        solver_utils.adjust_learning_rate(self.base_lr,
			self.optimizer)
            
    def update_lr(self):
        iters = self.iters
        if self.opt.TRAIN.LR_SCHEDULE == 'exp':
            solver_utils.adjust_learning_rate_exp(self.base_lr,
			self.optimizer, iters,
                        decay_rate=self.opt.EXP.LR_DECAY_RATE,
			decay_step=self.opt.EXP.LR_DECAY_STEP)

        elif self.opt.TRAIN.LR_SCHEDULE == 'inv':
            solver_utils.adjust_learning_rate_inv(self.base_lr, self.optimizer,
		    iters, self.opt.INV.ALPHA, self.opt.INV.BETA)
        elif self.opt.TRAIN.LR_SCHEDULE == 'fix':
            if self.loop+1 % self.opt.TRAIN.LR_UPDATE_LOOP == 0:
                solver_utils.adjust_learning_fix(self.base_lr, self.optimizer)
        else:
            raise NotImplementedError("Currently don't support the specified \
                    learning rate schedule: %s." % self.opt.TRAIN.LR_SCHEDULE)

    def log(self, target_accs):
        """
        Log results as JSON
        """
        save_path = self.opt.SAVE_DIR
        with open(os.path.join(save_path, 'perf_{}_{}.json'.format(self.source_name, self.target_name)), 'w') as f:
            json.dump(target_accs, f, indent=4)
            
    def logging(self, loss, accu):
        print('[loop: %d, iters: %d]: ' % (self.loop, self.iters))
        loss_names = ""
        loss_values = ""
        for key in loss:
            loss_names += key + ","
            loss_values += '%.4f,' % (loss[key])
        loss_names = loss_names[:-1] + ': '
        loss_values = loss_values[:-1] + ';'
        loss_str = loss_names + loss_values + (' source %s: %.4f.' %
                    (self.opt.EVAL_METRIC, accu))
        print(loss_str)

    def model_eval(self, preds, gts):
        assert(self.opt.EVAL_METRIC in ['mean_accu', 'accuracy']), \
             "Currently don't support the evaluation metric you specified."

        if self.opt.EVAL_METRIC == "mean_accu":
            res = mean_accuracy(preds, gts)
        elif self.opt.EVAL_METRIC == "accuracy":
            res = accuracy(preds, gts)
        return res

    def save_ckpt(self):
        save_path = self.opt.SAVE_DIR
        ckpt_resume = os.path.join(save_path, 'ckpt_final.resume')
        ckpt_weights = os.path.join(save_path, 'ckpt_final.weights')
        torch.save({'loop': self.loop,
                    'iters': self.iters,
                    'model_state_dict': self.net.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_resume)

        torch.save({'weights': self.net.module.state_dict(),
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_weights)

    def save_best_ckpt(self):
        save_path = self.opt.SAVE_DIR
        best_ckpt_resume = os.path.join(save_path, 'ckpt_best.resume')
        best_ckpt_weights = os.path.join(save_path, 'ckpt_best.weights')
        torch.save({'loop': self.loop,
                    'iters': self.iters,
                    'model_state_dict': self.net.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                    'bn_domain_map': self.bn_domain_map
                    }, best_ckpt_resume)

        torch.save({'weights': self.net.module.state_dict(),
                    'bn_domain_map': self.bn_domain_map
                    }, best_ckpt_weights)

    def complete_training(self):
        if self.loop > self.opt.TRAIN.MAX_LOOP:
            return True

    def register_history(self, key, value, history_len):
        if key not in self.history:
            self.history[key] = [value]
        else:
            self.history[key] += [value]

        if len(self.history[key]) > history_len:
            self.history[key] = \
                 self.history[key][len(self.history[key]) - history_len:]

    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample

    def get_samples_categorical(self, data_name, category):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader'][category]
        data_iterator = self.train_data[data_name]['iterator'][category]
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'][category] = data_iterator

        return sample

    def test(self):
        self.net.eval()
        preds = []
        gts = []
        for sample in iter(self.test_data['loader']):
            data, gt = to_cuda(sample['Img']), to_cuda(sample['Label'])
            logits = self.net(data)['logits']
            preds += [logits]
            gts += [gt]

        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)

        res = self.model_eval(preds, gts)
        return res

    def test_clustering(self):
        self.net.eval()
        preds = []
        gts = []
        for sample in iter(self.test_data['loader']):
            data, gt = to_cuda(sample['Img']), to_cuda(sample['Label'])
            feats = self.net(data)['feat']
            dist2center, labels = self.clustering.assign_labels(feats)
            preds += [labels]
            gts += [gt]

        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)

        res = 100.0 * torch.sum(gts == preds) / gts.size(0)
        return res
    
    def get_embedding(self, X, Y):
        train_transform = data_utils.get_transform(False)
        dataset_type = 'SingleDataset'
        dataset = getattr(single_dataset, dataset_type)(root="")
        dataset.initialize(root="",
                                transform=train_transform, classnames=self.classnames,
                                data_paths=X, data_labels=Y)
        loader_te = DataLoader(dataset,
                            shuffle=False, batch_size=self.opt.FINETUNE.TEST_BATCH_SIZE, num_workers=self.opt.FINETUNE.TEST_NUM_WORKERS)

        self.net.eval()
        embedding = torch.zeros([len(Y), self.net.module.in_dim])
        with torch.no_grad():
            for data in loader_te:
                x, y = to_cuda(data['Img']), to_cuda(data['Label'])
                feat = self.net(x)['feat']
                embedding[data['Index']] = feat.cpu()

        return embedding
    def get_grad_embedding(self, X, Y):
        train_transform = data_utils.get_transform(False)
        embDim = self.net.module.in_dim
        self.net.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        dataset_type = 'SingleDataset'
        dataset = getattr(single_dataset, dataset_type)(root="")
        dataset.initialize(root="",
                           transform=train_transform, classnames=self.classnames,
                           data_paths=X, data_labels=Y)
        loader_te = DataLoader(dataset,
                               shuffle=False, batch_size=self.opt.FINETUNE.TEST_BATCH_SIZE, num_workers=self.opt.FINETUNE.TEST_NUM_WORKERS)
        with torch.no_grad():
            for data in loader_te:
                x, y = to_cuda(data['Img']), to_cuda(data['Label'])
                cout, out = self.net(x)['probs'], self.net(x)['feat']
                out = out.cpu().numpy()
                batchProbs = cout.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[data['Index'][j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[data['Index'][j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
        return torch.Tensor(embedding)
    
    def predict_prob(self, X, Y):
        train_transform = data_utils.get_transform(False)
        dataset_type = 'SingleDataset'
        dataset = getattr(single_dataset, dataset_type)(root="")
        dataset.initialize(root="",
                                transform=train_transform, classnames=self.classnames,
                                data_paths=X, data_labels=Y)
        loader_te = DataLoader(dataset,
                            shuffle=False, batch_size=self.opt.FINETUNE.TEST_BATCH_SIZE, num_workers=self.opt.FINETUNE.TEST_NUM_WORKERS)

        self.net.eval()
        probs = torch.zeros([len(Y), self.opt.DATASET.NUM_CLASSES])
        with torch.no_grad():
            for data in loader_te:
                x, y = to_cuda(data['Img']), to_cuda(data['Label'])
                out = self.net(x)['probs']
                probs[data['Index']] = out.cpu()

        return probs
    
    def predict(self, X, Y):
        train_transform = data_utils.get_transform(False)
        dataset_type = 'SingleDataset'
        dataset = getattr(single_dataset, dataset_type)(root="")
        dataset.initialize(root="",
                                transform=train_transform, classnames=self.classnames,
                                data_paths=X, data_labels=Y)
        loader_te = DataLoader(dataset,
                            shuffle=False, batch_size=self.opt.FINETUNE.TEST_BATCH_SIZE, num_workers=self.opt.FINETUNE.TEST_NUM_WORKERS)

        self.net.eval()
        logits = torch.zeros([len(Y), self.opt.DATASET.NUM_CLASSES])
        with torch.no_grad():
            for data in loader_te:
                x, y = to_cuda(data['Img']), to_cuda(data['Label'])
                out = self.net(x)['logits']
                logits[data['Index']] = out.cpu()

        return logits
    
    def expected_calibration_error(self, y_true, y_pred, num_bins=15):
        pred_y = np.argmax(y_pred, axis=-1)
        correct = (pred_y == y_true).astype(np.float32)
        prob_y = np.max(y_pred, axis=-1)
        b = np.linspace(start=0, stop=1.0, num=num_bins)
        bins = np.digitize(prob_y, bins=b, right=True)
        o = 0
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += np.abs(np.sum(correct[mask] - prob_y[mask]))

        return o / y_pred.shape[0]
    
    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
        
    def random_sampling(self, n):
            idxs = np.random.choice(np.where(self.idxs_lb==0)[0], n)
            return idxs

    def query(self, n):
        if self.opt.FINETUNE.STRATEGY == "MarginSampling":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            probs_sorted, idxs = probs.sort(descending=True)
            U = probs_sorted[:, 0] - probs_sorted[:,1]
            idxs =idxs_unlabeled[U.sort()[1][:n]]
        elif self.opt.FINETUNE.STRATEGY == "RandomSampling":
            idxs = self.random_sampling(n)
        elif self.opt.FINETUNE.STRATEGY == "LeastConfidence":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            U = probs.max(1)[0]
            idxs = idxs_unlabeled[U.sort()[1][:n].cpu()]
        elif self.opt.FINETUNE.STRATEGY == "EntropySampling":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            log_probs = torch.log(probs)
            U = (probs*log_probs).sum(1)
            return idxs_unlabeled[U.sort()[1][:n].cpu()]
        elif self.opt.FINETUNE.STRATEGY == "Coreset":
            idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
            embedding = self.get_embedding(self.X, self.Y)
            embedding = embedding.numpy()
            sampling = kCenterGreedy(embedding) 
            idxs = sampling.select_batch_(idxs_labeled, n)
        elif self.opt.FINETUNE.STRATEGY == "KMeansSampling":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            embedding = embedding.numpy()
            cluster_learner = KMeans(n_clusters=n)
            cluster_learner.fit(embedding)

            cluster_idxs = cluster_learner.predict(embedding)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (embedding - centers)**2
            dis = dis.sum(axis=1)
            idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            idxs = idxs_unlabeled[idxs]
            tag = torch.ones(embedding.shape[0])
            visualize_2d("./img/tsne/"+self.opt.EXP_NAME, embedding, cluster_idxs, tag, embedding, cluster_idxs, self.opt.DATASET.NUM_CLASSES, str(self.loop))
        elif self.opt.FINETUNE.STRATEGY == "ClusterMarginSampling":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            probs_sorted, idxs = probs.sort(descending=True)
            U = (1-(probs_sorted[:, 0] - probs_sorted[:,1])) * 100 * 2
            
            #entropy = -torch.sum(probs * (torch.log(probs + 1e-5)), 1)
            #plt.scatter(entropy,U)
            #plt.xlabel("entropy")
            #plt.ylabel("margin")
            #plt.savefig('./img/scatter/'+self.opt.EXP_NAME+'/test_r'+str(self.loop)+'.png')
            #plt.close()
            
            embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            embedding = embedding.numpy()
            cluster_learner = KMeans(n_clusters=n)
            cluster_learner.fit(embedding, sample_weight=U)

            cluster_idxs = cluster_learner.predict(embedding)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (embedding - centers)**2
            dis = dis.sum(axis=1)
            idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            idxs = idxs_unlabeled[idxs]
            tag = torch.ones(embedding.shape[0])
            visualize_2d("./img/tsne/"+self.opt.EXP_NAME, embedding, cluster_idxs, tag, embedding, cluster_idxs, self.opt.DATASET.NUM_CLASSES, str(self.loop))
        elif self.opt.FINETUNE.STRATEGY == "ClusterEntropySampling":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            U = -torch.sum(probs * (torch.log(probs + 1e-5)), 1)
            
            #U = torch.pow(-torch.sum(probs * (torch.log(probs + 1e-5)), 1), 2*self.loop/self.opt.TRAIN.MAX_LOOP)
            U = -torch.sum(torch.pow(probs * (torch.log(probs + 1e-5)), 2*self.loop/self.opt.TRAIN.MAX_LOOP), 1)
            #probs_sorted, idxs2 = probs.sort(descending=True)
            #U2 = probs_sorted[:, 0] - probs_sorted[:,1]
            #plt.scatter(U,U2)
            #plt.xlabel("entropy")
            #plt.ylabel("margin")
            #plt.savefig('./img/scatter/'+self.opt.EXP_NAME+'/test_r'+str(self.loop)+'.png')
            #plt.close()
            
            embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            embedding = embedding.numpy()
            cluster_learner = KMeans(n_clusters=n)
            cluster_learner.fit(embedding, sample_weight=U)
            
            #if self.loop > self.opt.MME.MAX_EPOCH:
            #    cluster_learner.fit(embedding, sample_weight=U)
            #else:
            #    cluster_learner.fit(embedding)
            cluster_idxs = cluster_learner.predict(embedding)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (embedding - centers)**2
            dis = dis.sum(axis=1)
            idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            idxs = idxs_unlabeled[idxs]
            tag = torch.ones(embedding.shape[0])
            #visualize_2d("./img/tsne/"+self.opt.EXP_NAME, embedding, cluster_idxs, tag, embedding, cluster_idxs, self.opt.DATASET.NUM_CLASSES, str(self.loop))
        elif self.opt.FINETUNE.STRATEGY == "OurSampling":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])            
            probs_sorted, idxs = probs.sort(descending=True)
            U = probs_sorted[:, 0] - probs_sorted[:,1]
            embeddings = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            embeddings = embeddings.numpy()
            cluster_learner = KMeans(n_clusters=self.opt.DATASET.NUM_CLASSES)
            cluster_learner.fit(embeddings)
            
            cluster_idxs = cluster_learner.predict(embeddings)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (embeddings - centers)**2
            dis = dis.sum(axis=1)
            #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
            value = (1-U)* dis 
            
            chunk_size_list = np.add(np.array([int(n/self.opt.DATASET.NUM_CLASSES) for i in range(self.opt.DATASET.NUM_CLASSES)]), np.append(np.ones(n%self.opt.DATASET.NUM_CLASSES, np.int), np.zeros(self.opt.DATASET.NUM_CLASSES-n%self.opt.DATASET.NUM_CLASSES, np.int)))
            random.shuffle(chunk_size_list)
            #idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][U[cluster_idxs==i].sort()[1][:chunk_size_list[i]]] for i in range(self.opt.DATASET.NUM_CLASSES) if chunk_size_list[i]!=0])
            idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][value[cluster_idxs==i].sort()[1][:chunk_size_list[i]]] for i in range(self.opt.DATASET.NUM_CLASSES) if chunk_size_list[i]!=0])
            idxs = idxs_unlabeled[idxs]
        elif self.opt.FINETUNE.STRATEGY == "OurEntropySampling":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])            
            U = -torch.sum(probs * (torch.log(probs + 1e-5)), 1)
            embeddings = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            embeddings = embeddings.numpy()
            cluster_learner = KMeans(n_clusters=self.opt.DATASET.NUM_CLASSES)
            cluster_learner.fit(embeddings)
            
            cluster_idxs = cluster_learner.predict(embeddings)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (embeddings - centers)**2
            dis = dis.sum(axis=1)
            #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
            value = U* dis 
            
            chunk_size_list = np.add(np.array([int(n/self.opt.DATASET.NUM_CLASSES) for i in range(self.opt.DATASET.NUM_CLASSES)]), np.append(np.ones(n%self.opt.DATASET.NUM_CLASSES, np.int), np.zeros(self.opt.DATASET.NUM_CLASSES-n%self.opt.DATASET.NUM_CLASSES, np.int)))
            random.shuffle(chunk_size_list)
            #idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][U[cluster_idxs==i].sort()[1][:chunk_size_list[i]]] for i in range(self.opt.DATASET.NUM_CLASSES) if chunk_size_list[i]!=0])
            idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][value[cluster_idxs==i].sort()[1][:chunk_size_list[i]]] for i in range(self.opt.DATASET.NUM_CLASSES) if chunk_size_list[i]!=0])
            idxs = idxs_unlabeled[idxs]    
        elif self.opt.FINETUNE.STRATEGY == "Badge":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled]).numpy()
            cluster_learner = KMeans(n_clusters=n)
            cluster_learner.fit(gradEmbedding)

            cluster_idxs = cluster_learner.predict(gradEmbedding)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (gradEmbedding - centers)**2
            dis = dis.sum(axis=1)
            idxs = np.array([np.arange(gradEmbedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            idxs = idxs_unlabeled[idxs]
        elif self.opt.FINETUNE.STRATEGY == "Thresholding":
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            U = -torch.sum(probs * (torch.log(probs + 1e-5)), 1)
            
            #probs_sorted, idxs2 = probs.sort(descending=True)
            #U2 = probs_sorted[:, 0] - probs_sorted[:,1]
            #plt.scatter(U,U2)
            #plt.xlabel("entropy")
            #plt.ylabel("margin")
            #plt.savefig('./img/scatter/'+self.opt.EXP_NAME+'/test_r'+str(self.loop)+'.png')
            #plt.close()
            
            embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            embedding = embedding.numpy()
            cluster_learner = KMeans(n_clusters=n)
            #cluster_learner.fit(embedding, sample_weight=U)
            
            if self.loop/self.opt.MME.MAX_EPOCH > self.opt.FINETUNE.THRESHOLD:
                cluster_learner.fit(embedding, sample_weight=U)
            else:
                cluster_learner.fit(embedding)
            cluster_idxs = cluster_learner.predict(embedding)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (embedding - centers)**2
            dis = dis.sum(axis=1)
            idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            idxs = idxs_unlabeled[idxs]
            
            #preds = torch.max(probs, dim=1).indices.cpu()
            #is_correct = np.equal(preds, self.Y[idxs_unlabeled])
            make_model_diagrams("./img/reliable_diagram/"+self.opt.EXP_NAME, self.loop, probs, torch.Tensor(self.Y[idxs_unlabeled]))

            #self.expected_calibration_error(self.Y[idxs_unlabeled], probs.numpy())
            #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        return idxs


    def adentropy(self, data, lamda, eta=1.0):
        out_t1 = self.net(data, reverse=True, eta=eta)['probs']
        loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                                  (torch.log(out_t1 + 1e-5)), 1))
        return loss_adent

    def clear_history(self, key):
        if key in self.history:
            self.history[key].clear()

    def solve(self):
        pass

    def update_network(self, **kwargs):
        pass

