#### 手撕 BN
import torch
from torch import nn


def batch_norm(X, parameters, moving_mean, moving_var, eps, momentum):
    #### 预测模式下
    if not torch.is_grad_enable():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        return X_hat
    ### 训练模式下
    else:
        assert len(X.shape) in (2, 4)
        #### 全连接层
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        ### 卷积层
        elif len(X.shape) == 4:
            mean = X.mean(dim=(0, 2, 3))
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3))
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * moving_var
    Y = parameters['gamma'] * X_hat + parameters['beta']
    return Y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):

        super.__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.parameters = {}
        self.parameters['gamma'] = nn.parameters(torch.ones(shape))
        self.parameters['beta'] = nn.parametersa(torch.zeros(shape))
        self.moving_mean, self.moving_var = torch.ones(shape), torch.zeros(shape)


def forward(self, X):
    Y, self.moving_mean, self.moving_var = batch_norm(X, self.parameters, self.moving_mean, self.moving_var, eps=1e-5,
                                                      momentum=0.9)
    return Y

