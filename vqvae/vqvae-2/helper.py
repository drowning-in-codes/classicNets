import torch
import torch.nn as nn


class HelperModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HelperModule, self).__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_parameter_count(net: nn.Module):
        return sum(p.numel() for p in net.parameters())

    @staticmethod
    def get_device(cpu: bool):
        if cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
