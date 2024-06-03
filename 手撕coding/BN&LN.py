import numpy as np

class BatchNorm:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None

    def forward(self, x, mode='train'):
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        if mode == 'train':
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        elif mode == 'test':
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):
        pass

class LayerNorm:
    def __init__(self, epsilon=1e-5, gamma=None, beta=None):
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.params, self.grads = [], []

    def forward(self, x, mode='train'):
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):
        pass

# 测试示例
# 初始化批量归一化层
bn = BatchNorm()
# 初始化层归一化层
ln = LayerNorm()

# 假设输入是一个4维张量
input_data = np.random.randn(2, 3, 4, 5)

# 批量归一化
output_bn = bn.forward(input_data)
print("Batch Normalization output shape:", output_bn.shape)

# 层归一化
output_ln = ln.forward(input_data
