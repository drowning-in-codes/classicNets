import torch.nn as nn


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                                      1.0 - self.decay
                              ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    model = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU())
    ema = EMA(model, 0.99)
    ema.register()
    # training loop
    epochs = 20
    for i in range(epochs):
        # y = model(x)
        # train model
        ema.update
    ema.apply_shadow()
