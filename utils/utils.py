import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import itertools
import os
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
import scipy.io as scio
import math

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

def mean_accuracy(preds, target):
    num_classes = preds.size(1)
    preds = torch.max(preds, dim=1).indices
    accu_class = []
    for c in range(num_classes):
        mask = (target == c)
        c_count = torch.sum(mask).item()
        if c_count == 0: continue
        preds_c = torch.masked_select(preds, mask)
        accu_class += [1.0 * torch.sum(preds_c == c).item() / c_count]
    return 100.0 * np.mean(accu_class)

def accuracy(preds, target):
    preds = torch.max(preds, dim=1).indices
    return 100.0 * torch.sum(preds == target).item() / preds.size(0)

def save_preds(paths, preds, save_path, filename='preds.txt'):
    assert(len(paths) == preds.size(0))
    with open(os.path.join(save_path, filename), 'w') as f:
        for i in range(len(paths)):
            line = paths[i] + ' ' + str(preds[i].item()) + '\n'
            f.write(line)


def draw(net, bn_domain_map, dataloader, q_idxs, cfg, filename):
    #pca = PCA(n_components=2)

    # initialize model
    model_state_dict = None
    fx_pretrained = True

    # test
    res = {}
    res['path'], res['preds'], res['gt'], res['probs'] = [], [], [], []
    

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
                out = net(img)
                probs = out['probs']

            preds = torch.max(probs, dim=1).indices
            res['preds'] += [preds]
            res['probs'] += [probs]
            X_embedded_list.append(out['feat'].detach().cpu().numpy())
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
        # X_reduced = pca.fit_transform(X)
        Y = np.concatenate(Y_list)
        GT = np.concatenate(gt_list)
        classes = dataloader.classnames
        plt.figure(figsize=(10,10), dpi=80)
        cnf_matrix = confusion_matrix(GT, Y, labels=list(range(cfg.DATASET.NUM_CLASSES)))
        plot_confusion_matrix(cnf_matrix, classes=classes,normalize=True,
                                title=filename + ' all target data confusion matrix')
        plt.savefig('./img/confusion_matrix/'+cfg.EXP_NAME+'/'+filename+'_all_target_data_confusion_matrix.png')
        plt.close()
        plt.figure(figsize=(10,10), dpi=80)
        cnf_matrix = confusion_matrix(GT[q_idxs], Y[q_idxs], labels=list(range(cfg.DATASET.NUM_CLASSES)))
        plot_confusion_matrix(cnf_matrix, classes=classes,normalize=True,
                                title=filename + ' query data confusion matrix')
        plt.savefig('./img/confusion_matrix/'+cfg.EXP_NAME+'/'+filename+'_query_data_confusion_matrix.png')
        plt.close()
        '''
        levels = np.arange(0, cfg.DATASET.NUM_CLASSES, 1)
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
        plt.savefig('./img/decision_boundary/'+cfg.EXP_NAME+'/'+'_'+filename+'_decision_boundary.png')
        plt.close()
        '''
        # for i in range(cfg.DATASET.NUM_CLASSES-1):
            # for j in range(i+1,cfg.DATASET.NUM_CLASSES):
                # X_reduced_selected = X_reduced[(Y==i) | (Y==j)]
                # Y_selected = Y[(Y==i) | (Y==j)]
                # GT_selected = GT[(Y==i) | (Y==j)]
                # fig, ax = plt.subplots()
                # # Set-up grid for plotting.
                # X0, X1 = X_reduced_selected[:, 0], X_reduced_selected[:, 1]
                # xx, yy = make_meshgrid(X0, X1)
                # background_model = KNeighborsClassifier(n_neighbors=1).fit(X_reduced_selected, Y_selected)

                # out = plot_contours(ax, background_model, xx, yy, levels=levels, cmap=plt.cm.Spectral, alpha=0.8)
                # ax.scatter(X0, X1, c=GT_selected, cmap=plt.cm.Spectral, norm=cnorm, s=20, edgecolors='k')
                # ax.set_ylabel('PC2')
                # ax.set_xlabel('PC1')
                # # ax.set_xticks(())
                # # ax.set_yticks(())
                # ax.set_title(str(i)+' to '+str(j)+' Decison surface using the PCA transformed/projected features')
                # plt.savefig('./img/decision_boundary/'+cfg.EXP_NAME+'/'+str(i)+'_'+str(j)+'_'+filename+'_decision_boundary.png')
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

def save_matlab(data, label, save_path):
    # data contains t-SNE reduced 2-Dimension or 3-Dimension matrix, label is either "class" or "domain"
    print("save to matlab file")
    scio.savemat(save_path, {"x": np.transpose(
        data[:, 0], axes=0), "y": np.transpose(data[:, 1], axes=0), "label": np.transpose(label, axes=0)})
    print("mat file saved")


def visualize_2d(save_path, embedding, label, domain, tar_embedding, tar_label, class_num, filename):

    save_matlab(embedding, domain, os.path.join(
        save_path, "TSNE_Domain_2D.mat"))

    save_matlab(embedding, label, os.path.join(
        save_path, "TSNE_Label_2D.mat"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0.0, 1.0, class_num))

    xx = embedding[:, 0]
    yy = embedding[:, 1]
    
    for i in range(class_num):
        ax.scatter(xx[label == i], yy[label == i],
                   color=colors[i], s=10)

    for i in range(tar_embedding.shape[0]):
        ax.text(tar_embedding[i, 0], tar_embedding[i, 1], str(tar_label[i]), fontdict={"size": 10})
    

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(os.path.join(save_path, "TSNE_Label_2D_"+filename+".pdf"),
                format='pdf', dpi=600)
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0.0, 1.0, class_num))
    
    for i in range(2):
        ax.scatter(xx[domain == i], yy[domain == i], color=cm.bwr(i/1.), s=10)
    
    
    #for i in range(embedding.shape[0]):
    #    ax.text(xx[i], yy[i], str(int(domain[i])), fontdict={
    #             "size": 10}, color=cm.bwr(domain[i]/1.))
    

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(os.path.join(save_path, "TSNE_Domain_2D_"+filename+".pdf"),
                format='pdf', dpi=600)
    plt.show()
    plt.close()


# +
def draw_count_bar(save_path, count, class_num, filename):
    yint = range(min(count), math.ceil(max(count))+1) 
    xx = np.arange(class_num)
    plt.figure(figsize=(12,4))
    plt.bar(xx, count) 
    plt.yticks(yint)  
    plt.xticks(xx) 
    plt.xlabel('Ground Truth Classes') 
    plt.ylabel('Count') 
    plt.title('Label Histogram')
    plt.savefig(os.path.join(save_path, 'Label_Histogram_'+filename+'.png'))
    plt.close()

def visualize_3d(save_path, embedding, label, domain, tar_embedding, tar_label, class_num, filename):

    save_matlab(embedding, domain, os.path.join(
        save_path, "TSNE_Domain_3D.mat"))

    save_matlab(embedding, label, os.path.join(
        save_path, "TSNE_Label_3D.mat"))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.rainbow(np.linspace(0.0, 1.0, class_num))

    xx = embedding[:, 0]
    yy = embedding[:, 1]
    zz = embedding[:, 2]

    for i in range(class_num):
        ax.scatter(xx[label == i], yy[label == i],
                   zz[label == i], color=colors[i], s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(os.path.join(save_path, "TSNE_Label_3D.pdf"),
                format='pdf', dpi=600)
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(2):
        ax.scatter(xx[domain == i], yy[domain == i],
                   zz[domain == i], color=cm.bwr(i/1.), s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(os.path.join(save_path, "TSNE_Domain_3D.pdf"),
                format='pdf', dpi=600)
    plt.show()
    plt.close()


# -

def _calculate_ece(logits, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = logits
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def make_model_diagrams(save_path, loop, outputs, labels, n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    softmaxes = outputs
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions==labels).sum().item()/len(labels)
    
    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)
    
    plt.figure(0, figsize=(8, 8))
    gap = np.array(bin_scores - bin_corrects)
    
    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
    bin_corrects = np.nan_to_num(np.array([bin_correct  for bin_correct in bin_corrects]))
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')

    ece = _calculate_ece(outputs, labels)

    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    plt.text(0.17, 0.82, "ECE: {:.4f}".format(ece), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    plt.title("Reliability Diagram", size=22)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(os.path.join(save_path, "reliability_diagram_"+str(loop)+".png"))
    plt.show()
    plt.close()
    return ece
