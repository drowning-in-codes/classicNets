#  #!/usr/bin/env python
#  #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/12 下午4:49
#   lastModifiedTime:2024/6/12 下午4:49
#   file:utils.py
#   software: classicNets
#

import os
import random
import logging

import numpy as np
import torch


def set_seeds(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device():
    """get device (CPU or GPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {}, n_gpu: {}".format(device, n_gpu))
    return device


def split_last(x, shape):
    """split the last dimension to given shape"""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert 1 < n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def find_sublist(haystack, needle):
    """
    Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.
    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
            else:
                return i - n + 1
    return -1


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_random_word(vocab_words):
    i = random.randint(0, len(vocab_words) - 1)
    return vocab_words[i]


def get_logger(name, log_path):
    """get logger"""
    logger = logging.getLogger(name)
    fomatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not os.path.isfile(log_path):
        f = open(log_path, "w+")
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.DEBUG)
    return logger
