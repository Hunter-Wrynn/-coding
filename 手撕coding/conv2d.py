import numpy as np

def conv2d(inputs,kernels,padding,bias,stride):
    c,w,h=inputs.shape

    inputs = np.pad(inputs, ((0,0),(1,1),(1,1)))
    kernels_num,kernel_size=kernels.shape[0],kernels.shape[2]
    outputs=np.ones((kernels_num,(w-kernel_size+2*padding)//stride+1,(h-kernel_size+2*padding)//stride+1))
    for i in range(0, w - kernel_size + 2 * padding + 1, stride):
        for j in range(0, h - kernel_size + 2 * padding + 1, stride):
            outputs[:, i // stride, j // stride] = np.sum(
                np.multiply(kernels, inputs[:, i:i + kernel_size, j:j + kernel_size]), axis=(1, 2, 3)) + bias
    return outputs

