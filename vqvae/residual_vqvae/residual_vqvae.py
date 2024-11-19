import random
from math import ceil
from functools import partial, cache
from itertools import zip_longest

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.distributed as dist
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from einops import rearrange, repeat, reduce, pack, unpack
from einx import get_at
