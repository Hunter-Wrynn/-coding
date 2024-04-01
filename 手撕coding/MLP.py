#### 手撕 torch神经网络
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


### 定义模型
### N,1 -> N,10 -> N,10 -> N,1
class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(n_input, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


model = Net(1, 20, 1)
print(model)

### 准备数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(3) + 0.1 * torch.randn(x.size())

x, y = (Variable(x), Variable(y))

plt.scatter(x.data, y.data)
# 或者采用如下的方式也可以输出x,y
# plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

####  pipeline
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

for t in range(500):
    predict = model(x)
    loss = loss_func(predict, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.05)
