import torchvision.transforms as transforms
from PIL import Image
import torch
from config.config import cfg
import random
import os

# +
def get_transform(train=True):
    transform_list = []
    if cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'resize_and_crop':
        osize = [cfg.DATA_TRANSFORM.LOADSIZE, cfg.DATA_TRANSFORM.LOADSIZE]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        if train:
            transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
        else:
            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                transform_list.append(transforms.FiveCrop(cfg.DATA_TRANSFORM.FINESIZE)) 
            else:
                transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

    elif cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'crop':
        if train:
            transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
        else:
            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                transform_list.append(transforms.FiveCrop(cfg.DATA_TRANSFORM.FINESIZE)) 
            else:
                transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

    if train and cfg.DATA_TRANSFORM.FLIP:
        transform_list.append(transforms.RandomHorizontalFlip())

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize(mean=cfg.DATA_TRANSFORM.NORMALIZE_MEAN,
                                       std=cfg.DATA_TRANSFORM.NORMALIZE_STD)]

    if not train and cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
        transform_list += [transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose(to_normalized_tensor)(crop) for crop in crops]))]
    else:
        transform_list += to_normalized_tensor

    return transforms.Compose(transform_list)

def read_image_list(im_dir, n_max=None, n_repeat=None):
    items = []

    for imname in listdir_nohidden(im_dir):
        imname_noext = os.path.splitext(imname)[0]
        label = int(imname_noext.split('_')[1])
        impath = os.path.join(im_dir, imname)
        items.append((impath, label))

    if n_max is not None:
        items = random.sample(items, n_max)

    if n_repeat is not None:
        items *= n_repeat

    return items

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.
    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.')]
    if sort:
        items.sort()
    return items
