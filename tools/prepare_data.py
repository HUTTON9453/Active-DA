import os
import torch
import torch.utils.data as data
import data.utils as data_utils
from data import single_dataset
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from data.class_aware_dataset_dataloader import ClassAwareDataLoader
from data.image_folder import make_dataset_with_labels
from config.config import cfg
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

# +
def prepare_data_Logit(seed=None):
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)
    
    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source, "train_images")
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    
    classes = list(range(10))

    
    item_S = data_utils.read_image_list(dataroot_S)
    data_paths_S = [ item[0] for item in item_S ]
    data_labels_S = [ item[1] for item in item_S ]
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    
    X_train, X_test, y_train, y_test = train_test_split( data_paths_S, data_labels_S, 
                                                        test_size=0.025, random_state=seed, 
                                                        stratify=data_labels_S)
    dataloaders[source] = CustomDatasetDataLoader(dataset_root=dataroot_S, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=train_transform, 
                                                  data_paths=X_test, data_labels=y_test,
                                                  train=True, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    '''
    dataloaders[source] = CustomDatasetDataLoader(dataset_root=dataroot_S, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=train_transform, 
                                                  data_paths=data_paths_S, data_labels=data_labels_S,
                                                  train=True, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    '''
    item_T = data_utils.read_image_list(os.path.join(dataroot_T, "train_images"))
    data_paths_T = [ item[0] for item in item_T ]
    data_labels_T = [ item[1] for item in item_T ]
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % target)
    '''
    X_train, X_test, y_train, y_test = train_test_split( data_paths_T, data_labels_T, 
                                                        test_size=0.025, random_state=seed, 
                                                        stratify=data_labels_T)
    dataloaders[target] = CustomDatasetDataLoader(dataset_root=dataroot_T, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=train_transform, 
                                                  data_paths=X_test, data_labels=y_test,
                                                  train=True, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    
    '''
    dataloaders[target] = CustomDatasetDataLoader(dataset_root=dataroot_T, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=train_transform, 
                                                  data_paths=data_paths_T, data_labels=data_labels_T,
                                                  train=True, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    
    
    dataloaders[target+'_labeled'] = None
    
    item_test = data_utils.read_image_list(os.path.join(dataroot_T, "test_images"))
    data_paths_test = [ item[0] for item in item_test ]
    data_labels_test = [ item[1] for item in item_test ]
    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    print('Building %s test dataloader...' % test_domain)
    '''
    X_train, X_test, y_train, y_test = train_test_split( data_paths_test, data_labels_test, 
                                                        test_size=0.025, random_state=seed, 
                                                        stratify=data_labels_test)
    dataloaders['test'] = CustomDatasetDataLoader(dataset_root=dataroot_test, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=test_transform, 
                                                  data_paths=X_test, data_labels=y_test,
                                                  train=False, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    
    '''
    dataloaders['test'] = CustomDatasetDataLoader(dataset_root=dataroot_test, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=test_transform, 
                                                  data_paths=data_paths_test, data_labels=data_labels_test,
                                                  train=False, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    
    idxs_lb = np.zeros(len(data_labels_T), dtype=bool)
    
    print('number of labeled pool: {}'.format(len(dataloaders[source].dataset)))
    print('number of unlabeled pool: {}'.format(len(dataloaders[target].dataset)))
    print('number of testing pool: {}'.format(len(dataloaders['test'].dataset)))
    return dataloaders, data_paths_T, data_labels_T, idxs_lb, classes

def prepare_data_Office31(seed=None):
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)
    
    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    
    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)
    
    
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    dataloaders[source] = CustomDatasetDataLoader(dataset_root=dataroot_S, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=train_transform, 
                                                  train=True, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    # split target dataset
    target_data_paths, target_data_labels = make_dataset_with_labels(dataroot_T, classes)
    X_train, X_test, y_train, y_test = train_test_split( target_data_paths, target_data_labels, 
                                                        test_size=cfg.TEST.TEST_PERCENTAGE, random_state=seed, 
                                                        stratify=target_data_labels)
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % target)
    dataloaders[target] = CustomDatasetDataLoader(dataset_root=dataroot_T, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=train_transform, 
                                                  data_paths=X_train, data_labels=y_train,
                                                  train=True, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    
    dataloaders[target+'_labeled'] = None
    
    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    print('Building %s test dataloader...' % test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(dataset_root=dataroot_test, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=test_transform, 
                                                  data_paths=X_test, data_labels=y_test,
                                                  train=False, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)

    idxs_lb = np.zeros(len(y_train), dtype=bool)
    
    print('number of labeled pool: {}'.format(len(dataloaders[source].dataset)))
    print('number of unlabeled pool: {}'.format(len(dataloaders[target].dataset)))
    print('number of testing pool: {}'.format(len(dataloaders['test'].dataset)))
    return dataloaders, X_train, y_train, idxs_lb, classes
    
    

def prepare_data_DomainNet(seed=None):
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)
    
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    dataloaders[source] = CustomDatasetDataLoader(dataset_root=dataroot_S, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=train_transform, 
                                                  train=True, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)
    
    with open(os.path.join(dataroot_T, target+'_train.txt'), 'r') as f:
        image_index = [os.path.join(cfg.DATASET.DATAROOT, x.split(' ')[0]) for x in f.readlines()]
    with open(os.path.join(dataroot_T, target+'_train.txt'), 'r') as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        y_train = np.array(label_list)
    X_train = image_index[selected_list]
    
    
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % target)
    dataloaders[target] = CustomDatasetDataLoader(dataset_root=dataroot_T, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=train_transform, 
                                                  data_paths=X_train, data_labels=y_train,
                                                  train=True, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)

    
    with open(os.path.join(dataroot_T, target+'_test.txt'), 'r') as f:
        image_index = [os.path.join(cfg.DATASET.DATAROOT, x.split(' ')[0]) for x in f.readlines()]
    with open(os.path.join(dataroot_T, target+'_test.txt'), 'r') as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        y_test = np.array(label_list)
    X_test = image_index[selected_list]
    
    dataloaders[target+'_labeled'] = None
    
    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    print('Building %s test dataloader...' % test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(dataset_root=dataroot_test, dataset_type=dataset_type, 
                                                  batch_size=batch_size, transform=test_transform, 
                                                  data_paths=X_test, data_labels=y_test,
                                                  train=False, num_workers=cfg.NUM_WORKERS, 
                                                  classnames=classes)

    print('number of labeled pool: {}'.format(len(dataloaders[source].dataset)))
    print('number of unlabeled pool: {}'.format(len(dataloaders[target].dataset)))
    print('number of testing pool: {}'.format(len(dataloaders['test'].dataset)))

    # generate initial labeled pool
    idxs_lb = np.zeros(len(y_train), dtype=bool)
    
    
    return dataloaders, X_train, y_train, idxs_lb, classes

def prepare_data_AADA(seed=None):
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)


    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    target_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_dataset.initialize(root=dataroot_T, classnames=classes)
    X_train, X_test, y_train, y_test = train_test_split( target_dataset.data_paths, target_dataset.data_labels, test_size=0.33, random_state=0, stratify=target_dataset.data_labels)
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDataset'
    target_test_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_test_dataset.initialize_data(data_paths=X_test, data_labels=y_test, transform=test_transform)
    dataset_type = 'SingleDataset'
    target_train_dataset = getattr(single_dataset, dataset_type)()
    target_train_dataset.initialize_data(data_paths=X_train, data_labels=y_train, transform=train_transform)

    dataloaders[target] = CustomDatasetDataLoader2(
                dataset=target_train_dataset, batch_size=batch_size,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)
    
    dataloaders[target+'_label'] = None

    source_x_train, source_y_train = make_dataset_with_labels(dataroot_S, classes)
    X_tr = [*source_x_train, *X_train]
    Y_tr = [*source_y_train, *y_train]
    # start experiment
    NUM_INIT_LB = len(source_y_train)
    n_pool = len(Y_tr)
    n_test = len(y_test)
    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))
    X_tr = X_train
    Y_tr = y_train

    # generate initial labeled pool
    idxs_lb = np.zeros(len(y_train), dtype=bool)
    handler = get_handler()

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader2(
                    dataset=target_test_dataset, batch_size=batch_size,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders, X_tr, Y_tr, idxs_lb, handler, train_transform, test_transform
def prepare_data_ACAN_improve_clustering(seed=None):
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    # for clustering
    batch_size = cfg.CLUSTERING.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building clustering_%s dataloader...' % source)
    dataloaders['clustering_' + source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.CLUSTERING.TARGET_BATCH_SIZE
    print('Building clustering_%s dataloader...' % target)
    # dataset_type = cfg.CLUSTERING.TARGET_DATASET_TYPE
    dataset_type = 'SingleDataset'
    # Random split
    target_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_dataset.initialize(root=dataroot_T, classnames=classes)
    X_train, X_test, y_train, y_test = train_test_split( target_dataset.data_paths, target_dataset.data_labels, test_size=0.33, random_state=0, stratify=target_dataset.data_labels)
    dataset_type = 'SingleDataset'
    target_test_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_test_dataset.initialize_data(data_paths=X_test, data_labels=y_test, transform=test_transform)
    # dataset_type = 'SingleDatasetWithoutLabel'
    dataset_type = 'SingleDataset'
    target_train_dataset = getattr(single_dataset, dataset_type)()
    target_train_dataset.initialize_data(data_paths=X_train, data_labels=y_train, transform=train_transform)

    dataloaders['clustering_' + target] = CustomDatasetDataLoader2(
                dataset=target_train_dataset, batch_size=batch_size,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    source_x_train, source_y_train = make_dataset_with_labels(dataroot_S, classes)
    X_tr = [*source_x_train, *X_train]
    Y_tr = [*source_y_train, *y_train]
    # start experiment
    NUM_INIT_LB = len(source_y_train)
    n_pool = len(Y_tr)
    n_test = len(y_test)
    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))
    X_tr = X_train
    Y_tr = y_train

    # generate initial labeled pool
    idxs_lb = np.zeros(len(y_train), dtype=bool)
    handler = get_handler()

    # class-agnostic source dataloader for supervised training
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    # initialize the categorical dataloader
    dataset_type = 'CategoricalSTDataset'
    source_batch_size = cfg.TRAIN.SOURCE_CLASS_BATCH_SIZE
    target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical'] = ClassAwareDataLoader(
                dataset_type=dataset_type,
                source_batch_size=source_batch_size,
                target_batch_size=target_batch_size,
                source_dataset_root=dataroot_S,
                transform=train_transform,
                classnames=classes,
                num_workers=cfg.NUM_WORKERS, seed=seed,
                drop_last=True, sampler='RandomSampler')

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader2(
                    dataset=target_test_dataset, batch_size=batch_size,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders, X_tr, Y_tr, idxs_lb, handler, train_transform, test_transform


# -

def prepare_data_ACAN(seed=None):
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    # for clustering
    batch_size = cfg.CLUSTERING.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building clustering_%s dataloader...' % source)
    dataloaders['clustering_' + source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.CLUSTERING.TARGET_BATCH_SIZE
    print('Building clustering_%s dataloader...' % target)
    # dataset_type = cfg.CLUSTERING.TARGET_DATASET_TYPE
    dataset_type = 'SingleDataset'
    # Random split
    target_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_dataset.initialize(root=dataroot_T, classnames=classes)
    X_train, X_test, y_train, y_test = train_test_split( target_dataset.data_paths, target_dataset.data_labels, test_size=0.33, random_state=0, stratify=target_dataset.data_labels)
    dataset_type = 'SingleDataset'
    target_test_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_test_dataset.initialize_data(data_paths=X_test, data_labels=y_test, transform=test_transform)
    # dataset_type = 'SingleDatasetWithoutLabel'
    dataset_type = 'SingleDataset'
    target_train_dataset = getattr(single_dataset, dataset_type)()
    target_train_dataset.initialize_data(data_paths=X_train, data_labels=y_train, transform=train_transform)

    dataloaders['clustering_' + target] = CustomDatasetDataLoader2(
                dataset=target_train_dataset, batch_size=batch_size,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)
    # dataloaders['clustering_' + target] = CustomDatasetDataLoader(
                # dataset_root=dataroot_T, dataset_type=dataset_type,
                # batch_size=batch_size, transform=train_transform,
                # train=False, num_workers=cfg.NUM_WORKERS,
                # classnames=classes)

    source_x_train, source_y_train = make_dataset_with_labels(dataroot_S, classes)
    X_tr = [*source_x_train, *X_train]
    Y_tr = [*source_y_train, *y_train]
    # start experiment
    NUM_INIT_LB = len(source_y_train)
    n_pool = len(Y_tr)
    n_test = len(y_test)
    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))
    X_tr = X_train
    Y_tr = y_train

    # generate initial labeled pool
    # idxs_lb = np.zeros(n_pool, dtype=bool)
    # idxs_lb[:NUM_INIT_LB] = True
    idxs_lb = np.zeros(len(y_train), dtype=bool)
    # idxs_lb[:n_pool] = True
    handler = get_handler()

    # class-agnostic source dataloader for supervised training
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    # initialize the categorical dataloader
    dataset_type = 'CategoricalSTDataset'
    source_batch_size = cfg.TRAIN.SOURCE_CLASS_BATCH_SIZE
    target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical'] = ClassAwareDataLoader(
                dataset_type=dataset_type,
                source_batch_size=source_batch_size,
                target_batch_size=target_batch_size,
                source_dataset_root=dataroot_S,
                transform=train_transform,
                classnames=classes,
                num_workers=cfg.NUM_WORKERS, seed=seed,
                drop_last=True, sampler='RandomSampler')

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader2(
                    dataset=target_test_dataset, batch_size=batch_size,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders, X_tr, Y_tr, idxs_lb, handler, train_transform, test_transform

def prepare_data_CAN():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    # for clustering
    batch_size = cfg.CLUSTERING.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building clustering_%s dataloader...' % source)
    dataloaders['clustering_' + source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.CLUSTERING.TARGET_BATCH_SIZE
    print('Building clustering_%s dataloader...' % target)
    # dataset_type = cfg.CLUSTERING.TARGET_DATASET_TYPE
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

    dataloaders['clustering_' + target] = CustomDatasetDataLoader2(
                dataset=target_train_dataset, batch_size=batch_size,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)
    # dataloaders['clustering_' + target] = CustomDatasetDataLoader(
                # dataset_root=dataroot_T, dataset_type=dataset_type,
                # batch_size=batch_size, transform=train_transform,
                # train=False, num_workers=cfg.NUM_WORKERS,
                # classnames=classes)

    # generate initial labeled pool
    n_pool = len(y_train)
    idxs_lb = np.zeros(n_pool, dtype=bool)

    # class-agnostic source dataloader for supervised training
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    # initialize the categorical dataloader
    dataset_type = 'CategoricalSTDataset'
    source_batch_size = cfg.TRAIN.SOURCE_CLASS_BATCH_SIZE
    target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical'] = ClassAwareDataLoader(
                dataset_type=dataset_type,
                source_batch_size=source_batch_size,
                target_batch_size=target_batch_size,
                source_dataset_root=dataroot_S,
                transform=train_transform,
                classnames=classes,
                num_workers=cfg.NUM_WORKERS,
                drop_last=True, sampler='RandomSampler')

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader2(
                    dataset=target_test_dataset, batch_size=batch_size,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders

def prepare_data_finetune():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)
    dataset_type = 'SingleDataset'
    # Random split
    target_dataset = getattr(single_dataset, dataset_type)(root=dataroot_T)
    target_dataset.initialize(root=dataroot_T, classnames=classes)
    target_x_train, X_te, target_y_train, Y_te = train_test_split( target_dataset.data_paths, target_dataset.data_labels, test_size=0.33, random_state=0, stratify=target_dataset.data_labels)
    source_x_train, source_y_train = make_dataset_with_labels(dataroot_S, classes)
    X_tr = [*source_x_train, *target_x_train]
    Y_tr = [*source_y_train, *target_y_train]
    # start experiment
    NUM_INIT_LB = len(source_y_train)
    n_pool = len(Y_tr)
    n_test = len(Y_te)
    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))

    # generate initial labeled pool
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_lb[:NUM_INIT_LB] = True
    # idxs_lb[:n_pool] = True
    handler = get_handler()

    return X_tr, Y_tr, np.array(X_te), torch.from_numpy(np.array(Y_te)), idxs_lb, handler, train_transform, test_transform


def prepare_data_MMD():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDatasetWithoutLabel'
    dataloaders[target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders

def prepare_data_SingleDomainSource():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders

def prepare_data_SingleDomainTarget():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    target = cfg.DATASET.TARGET_NAME
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDataset'
    dataloaders[target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders
