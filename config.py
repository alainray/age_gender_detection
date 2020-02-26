from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 4

# Preprocess options
__C.PREPROCESS = edict()
__C.PREPROCESS.LABEL_PATH = 'imdb/imdb.mat'
__C.PREPROCESS.FEATURES = ['features.pth']
__C.PREPROCESS = dict(__C.PREPROCESS)

# Training options
__C.TRAIN = edict()
__C.TRAIN.OPTIMIZER = 'adam'
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.00001
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.MAX_EPOCHS = 20
__C.TRAIN.WEIGHT_INIT = "xavier_uniform"
__C.TRAIN.CLIP_GRADS = True
__C.TRAIN.CLIP = 8
__C.TRAIN.EARLY_STOPPING = True
__C.TRAIN.PATIENCE = 2
__C.TRAIN.TRAIN_RATIO = 0.8
__C.TRAIN = dict(__C.TRAIN)

# Test
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 64
__C.TEST = dict(__C.TEST)

# Model options
__C.MODEL = edict()
__C.MODEL.AGE_LAMBDA = 0.5
__C.MODEL = dict(__C.MODEL)
__C.MODEL.LOAD_CHECKPOINT = ''
__C.MODEL.WEIGHT_DECAY = 0.0
__C.MODEL.AGE_NUM_CLASSES = 4

# Dataset options
__C.DATASET = edict()
__C.DATASET.DATA_FOLDER = ['storage/imdb_crop','storage/fairface']
__C.DATASET = dict(__C.DATASET)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif b[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
