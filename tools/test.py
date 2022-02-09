import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
from model import model
import data.utils as data_utils
from utils.utils import to_cuda, mean_accuracy, accuracy
from data import single_dataset
from data.custom_dataset_dataloader import CustomDatasetDataLoader, CustomDatasetDataLoader2
import sys
import pprint
from config.config import cfg, cfg_from_file, cfg_from_list
from math import ceil as ceil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import itertools
import random

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--adapted', dest='adapted_model',
                        action='store_true',
                        help='if the model is adapted on target')
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name',
                        default='exp', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--seed', dest='seed',
                        help='seed',
                        default=1126, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def save_preds(paths, preds, save_path, filename='preds.txt'):
    assert(len(paths) == preds.size(0))
    with open(os.path.join(save_path, filename), 'w') as f:
        for i in range(len(paths)):
            line = paths[i] + ' ' + str(preds[i].item()) + '\n'
            f.write(line)

def prepare_data():
    test_transform = data_utils.get_transform(False)
    train_transform = data_utils.get_transform(True)

    target = cfg.TEST.DOMAIN
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    dataloader = None

    dataset_type = 'SingleDataset'
    # Random split
    target_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_dataset.initialize(root=dataroot_T, classnames=classes)
    X_train, X_test, y_train, y_test = train_test_split( target_dataset.data_paths, target_dataset.data_labels, test_size=0.33, random_state=0, stratify=target_dataset.data_labels)
    dataset_type = 'SingleDataset'
    target_test_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_test_dataset.initialize_data(data_paths=X_test, data_labels=y_test, transform=test_transform)
    dataset_type = 'SingleDataset'
    target_train_dataset = getattr(single_dataset, dataset_type)()
    target_train_dataset.initialize_data(data_paths=X_train, data_labels=y_train, transform=train_transform)
    # dataset_type = cfg.TEST.DATASET_TYPE
    batch_size = cfg.TEST.BATCH_SIZE
	# train data
    dataloader = CustomDatasetDataLoader2(
                    dataset=target_train_dataset, batch_size=batch_size,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)
	# test data
    # dataloader = CustomDatasetDataLoader2(
                    # dataset=target_train_dataset, batch_size=batch_size,
                    # train=False, num_workers=cfg.NUM_WORKERS,
                    # classnames=classes)
    # dataloader = CustomDatasetDataLoader(dataset_root=dataroot_T,
                # dataset_type=dataset_type, batch_size=batch_size,
                # transform=test_transform, train=False,
                # num_workers=cfg.NUM_WORKERS, classnames=classes)

    return dataloader

def test(args):
    pca = PCA(n_components=2)
    # prepare data
    dataloader = prepare_data()

    # initialize model
    model_state_dict = None
    fx_pretrained = True

    bn_domain_map = {}
    if cfg.WEIGHTS != '':
        weights_dict = torch.load(cfg.WEIGHTS)
        model_state_dict = weights_dict['weights']
        bn_domain_map = weights_dict['bn_domain_map']
        fx_pretrained = False

    if args.adapted_model:
        num_domains_bn = 2
    else:
        num_domains_bn = 1

    net = model.danet(num_classes=cfg.DATASET.NUM_CLASSES,
                 state_dict=model_state_dict,
                 feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR,
                 fx_pretrained=fx_pretrained,
                 dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                 fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                 num_domains_bn=num_domains_bn)

    net = torch.nn.DataParallel(net)

    if torch.cuda.is_available():
        net.cuda()

    # test
    res = {}
    res['path'], res['preds'], res['gt'], res['probs'] = [], [], [], []
    net.eval()

    if cfg.TEST.DOMAIN in bn_domain_map:
        domain_id = bn_domain_map[cfg.TEST.DOMAIN]
    else:
        domain_id = 0

    X_embedded_list = []
    Y_list = []
    gt_list = []
    with torch.no_grad():
        net.module.set_bn_domain(domain_id)
        for sample in iter(dataloader):
            res['path'] += sample['Path']

            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                n, ncrop, c, h, w = sample['Img'].size()
                sample['Img'] = sample['Img'].view(-1, c, h, w)
                img = to_cuda(sample['Img'])
                probs = net(img)['probs']
                probs = probs.view(n, ncrop, -1).mean(dim=1)
            else:
                img = to_cuda(sample['Img'])
                probs = net(img)['probs']

            preds = torch.max(probs, dim=1)[1]
            res['preds'] += [preds]
            res['probs'] += [probs]
            X_embedded_list.append(net(img)['feat'].detach().cpu().numpy())
            Y_list.append(preds.detach().cpu().numpy())

            if 'Label' in sample:
                gt_list.append(sample['Label'].numpy())
                label = to_cuda(sample['Label'])
                res['gt'] += [label]
            print('Processed %d samples.' % len(res['path']))

        preds = torch.cat(res['preds'], dim=0)
        save_preds(res['path'], preds, cfg.SAVE_DIR)

        if 'gt' in res and len(res['gt']) > 0:
            gts = torch.cat(res['gt'], dim=0)
            probs = torch.cat(res['probs'], dim=0)

            assert(cfg.EVAL_METRIC == 'mean_accu' or cfg.EVAL_METRIC == 'accuracy')
            if cfg.EVAL_METRIC == "mean_accu":
                eval_res = mean_accuracy(probs, gts)
                print('Test mean_accu: %.4f' % (eval_res))

            elif cfg.EVAL_METRIC == "accuracy":
                eval_res = accuracy(probs, gts)
                print('Test accuracy: %.4f' % (eval_res))

        X = np.concatenate(X_embedded_list)
        X_reduced = pca.fit_transform(X)
        Y = np.concatenate(Y_list)
        GT = np.concatenate(gt_list)
        classes = dataloader.classnames
        plt.figure(figsize=(10,10), dpi=80)
        cnf_matrix = confusion_matrix(GT, Y)
        plot_confusion_matrix(cnf_matrix, classes=classes,normalize=True,
                                title=cfg.EXP_NAME + ' confusion matrix')
        plt.savefig('./img/confusion_matrix/'+cfg.EXP_NAME+'/'+'confusion_matrix.png')
        plt.close()
        levels = np.arange(0, 32, 1)
        cnorm = plt.Normalize(vmin=levels[0],vmax=levels[-1])
        clevels = [levels[0]] + list((levels[1:]+levels[:-1])) + [levels[-1]]
        colors=plt.cm.Spectral(cnorm(clevels))
        cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend='both')

        fig, ax = plt.subplots()
        # Set-up grid for plotting.
        X0, X1 = X_reduced[:, 0], X_reduced[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        background_model = KNeighborsClassifier(n_neighbors=1).fit(X_reduced, Y)

        out = plot_contours(ax, background_model, xx, yy, levels=levels, cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(X0, X1, c=GT, cmap=plt.cm.Spectral, norm=cnorm, s=20, edgecolors='k')
        ax.set_ylabel('PC2')
        ax.set_xlabel('PC1')
        ax.set_title(' Decison surface using the PCA transformed/projected features')
        plt.savefig('./img/decision_boundary/'+cfg.EXP_NAME+'/' + 'decision_boundary.png')
        plt.close()

        # for i in range(cfg.DATASET.NUM_CLASSES-1):
            # for j in range(i+1,cfg.DATASET.NUM_CLASSES):
                # X_reduced_selected = X_reduced[(Y==i) | (Y==j)]
                # Y_selected = Y[(Y==i) | (Y==j)]
                # GT_selected = GT[(Y==i) | (Y==j)]
                # fig, ax = plt.subplots()
                # # title for the plots
                # title = ('Decision surface of CAN with ground truth ')
                # # Set-up grid for plotting.
                # X0, X1 = X_reduced_selected[:, 0], X_reduced_selected[:, 1]
                # xx, yy = make_meshgrid(X0, X1)
                # background_model = KNeighborsClassifier(n_neighbors=1).fit(X_reduced_selected, Y_selected)

                # plot_contours(ax, background_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
                # ax.scatter(X0, X1, c=GT_selected, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                # out = plot_contours(ax, background_model, xx, yy, levels=levels, cmap=plt.cm.Spectral, alpha=0.8)
                # ax.scatter(X0, X1, c=GT_selected, cmap=plt.cm.Spectral, norm=cnorm, s=20, edgecolors='k')
                # ax.set_ylabel('PC2')
                # ax.set_xlabel('PC1')
                # ax.set_xticks(())
                # ax.set_yticks(())
                # ax.set_title(str(i)+' to '+str(j)+' Decison surface using the PCA transformed/projected features')
                # plt.savefig('./img/decision_boundary/'+cfg.EXP_NAME+'/'+str(i)+'_'+str(j)+'_decision_boundary.png')
                # plt.close()
                # print(str(i)+'_'+str(j)+'_'+'decision_boundary.png finished')

    print('Finished!')

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt),
                 # horizontalalignment="center",
                 # color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.weights is not None:
        cfg.WEIGHTS = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name

    print('initial seed:' + str(args.seed))
    seed_torch(args.seed)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    test(args)
