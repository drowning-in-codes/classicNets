#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/13 下午10:19
#   lastModifiedTime:2024/6/13 下午10:19
#   file:train.py
#   software: classicNets
#

import os
import json
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn


class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431  # random seed
    batch_size: int = 32
    lr: int = 5e-5  # learning rate
    n_epochs: int = 10  # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100  # interval for saving model
    total_steps: int = 100000  # total number of steps to train

    @classmethod
    def from_json(cls, file):  # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer:
    def __init__(self, cfg, model, date_iter, optimizer, save_dir, device):
        self.cfg = cfg
        self.model = model
        self.data_iter = date_iter
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device

    def train(self, get_loss, model_file=None, pretrain_file=None, data_parallel=True):
        """ train loop """
        self.model.train()
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.Parallel(model)
        global_step = 0
        for e in range(self.cfg.n_epochs):
            loss_sum = 0
            iter_bar = tqdm(self.data_iter, desc="Iter (loss=X.XXX)")
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                loss = get_loss(model, batch, global_step).mean()
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.sum()
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
                if global_step % self.cfg.save_steps == 0:
                    self.save(global_step)
                if self.cfg_total_steps and self.cfg.total_steps <= global_step:
                    print("Epoch %d/%d:Average Loss %5.3f" % (e + 1, self.cfg.n_epochs, loss_sum / (i + 1),))
                    print("The Total steps have been reached")
                    self.save(global_step)
                    return
                print('Epoch %d/%d : Average Loss %5.3f' % (e + 1, self.cfg.n_epochs, loss_sum / (i + 1)))
            self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        self.model.eval()
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        results = []
        iter_bar = tqdm(self.data_iter, desc="Iter (loss=X.XXX)")
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # evaluation without gradient calculation
                accuracy, result = evaluate(model, batch)  # accuracy to print
            results.append(result)
            iter_bar.set_description('Iter(acc=%5.3f)' % accuracy)
        return results

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file:  # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            self.model.load_state_dict(torch.load(pretrain_file), strict=False)

    def train(self, i):
        torch.save(self.model.state_dict(), str(os.path.join(self.save_dir, "model_%d.pth" % i)))
