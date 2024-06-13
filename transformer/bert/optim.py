#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/13 下午8:19
#   lastModifiedTime:2024/6/13 下午8:19
#   file:optim.py
#   software: classicNets
#
import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_


def warmup_cosine(x, warmup=.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_costant(x, warmup=.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


SCHEDULES = {
    "warmup_cosine": warmup_cosine,
    "warmup_constant": warmup_costant,
    "warmup_linear": warmup_linear
}


class BertAdam(Optimizer):
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule="warmup_linear", b1=0.9, b2=0.009, e=1e-6,
                 weight_decay_rate=.01, max_grad_norm=1.0):
        assert lr > .0, "Learning rate:%f -shoule be >0.0" % (lr)
        assert schedule in SCHEDULES, "Invalid schedule:%s" % (schedule)
        assert .0 <= warmup < 1.0 or warmup == -1, \
            "Warmup %f - shoule be in 0.0~1.0 or -1 (no warm up)" % (warmup)
        assert 0.0 <= b1 < 1.0, "b1:%f - should be in 0.0~1.0" % (b1)
        assert .0 <= b1 < 1.0, "b2:%f - should be in 0.0~1.0" % (b2)
        assert e > .0, "epsilon:%f - should be >0.0" % (e)
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total, b1=b1, b2=b2, e=e,
                        weight_decay_rate=weight_decay_rate, max_grad_norm=max_grad_norm)

        super().__init__(params, defaults)

    def get_lr(self):
        """ get learning rate in training """
        lr = []
        for group in self.param_groups:
            """
    aram_groups是一个列表,其中每个元素都是一个字典,包含了某些参数的优化配置。每个字典都有以下键:
    params: 一个包含这组参数的列表。
    lr: 这组参数的学习率。
    momentum: 这组参数的动量参数。
    weight_decay: 这组参数的权重衰减参数。
    其他一些可选的键,如dampening、centered等,具体取决于所使用的优化器
            """
            for p in group["params"]:
                state = self.state[p]
                """
                存储了优化器的状态信息。
                state是一个嵌套的字典
                
                state字典中存储了优化器在训练过程中需要维护的一些中间状态,比如:
                动量缓冲区: 用于实现动量优化的缓冲区。
                指数移动平均: 用于实现 Adam 优化器的指数移动平均。
                指数移动平均平方: 用于实现 Adam 优化器的指数移动平均平方
                """
                if not state:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support gradients,please consider SparseAdam instead")
                state = self.state[p]
                # state initialization
                if not state:
                    # 如果没有state,则初始化state
                    state['step'] = 0
                    state['next_m'] = torch.zeros_like(p.data)
                    state['next_v'] = torch.zeros_like(p.data)
                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2'],
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update

                p.data.add_(-update_with_lr)
                state['step'] += 1
        return loss


def optim4GPU(cfg, model):
    """ optimizer for GPU training """
    param_optimizer = list(model.named_paramers())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': .0}
    ]
    return BertAdam(optimizer_grouped_parameters, lr=cfg.lr, warmup=cfg.warmup, t_total=cfg.total_steps)
