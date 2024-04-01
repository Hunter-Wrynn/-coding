
import numpy as np

def pooling(inputs, pool_size, stride, mode='max'):
    c, w, h = inputs.shape
    k = pool_size
    outputs = np.zeros((c, (w - k) // stride + 1, (h - k) // stride + 1))
    if mode == 'max':
        for i in range(0, w - k + 1, stride):
            for j in range(0, h - k + 1, stride):
                outputs[:, i // stride, j // stride] = np.max(inputs[:, i:i + k, j:j + k], axis=(1, 2))
        return outputs
    elif mode == 'avg':
        for i in range(0, w - k + 1, stride):
            for j in range(0, h - k + 1, stride):
                outputs[:, i // stride, j // stride] = np.mean(inputs[:, i:i + k, j:j + k], axis=(1, 2))
        return outputs
    else:
        raise ValueError('not support this mode, choose "max" or "avg" ')


pool_size = 2
stride = 2
mode = 'max'
inputs = np.arange(1, 76).reshape((3, 5, 5))
print("inputs:{}".format(inputs.shape), '\n', inputs)
outputs = pooling(inputs, pool_size, stride, mode)
print("outputs:{}".format(outputs.shape), '\n', outputs)
