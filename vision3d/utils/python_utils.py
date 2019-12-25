import os
import os.path as osp


def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path)
