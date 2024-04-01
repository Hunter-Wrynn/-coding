# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import numpy as np
from numpy.random import randn
from math import sqrt
import torch
import torch.nn as nn

def softmax(x):
    e_x=np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=0)

d=256
n=32
x=randn(d,n)

w_q=randn(d,d)
w_k=randn(d,d)
w_v=randn(d,d)

q=w_q@x
k=w_k@x
v=w_v@x

A=k.T@q

A/=sqrt(d)
A_hat=softmax(A)
output=v@A_hat

class self_attention(nn.Module):
    def __init__(self,input_dim,dim_k,dim_v):
        super(self_attention, self).__init__()
        self.q=nn.linear(input_dim,dim_k)
        self.k = nn.linear(input_dim, dim_k)
        self.v = nn.linear(input_dim, dim_v)
        self.norm =1/sqrt(dim_k)

    def forward(self,x):
        Q=self.q(x)
        K=self.k(x)
        V=self.v(x)

        atten=nn.Softmax(dim=-1)(torch.bnn(Q,K.permute(0,2,1))) *self.norm

        return output