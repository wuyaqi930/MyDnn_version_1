import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np  


#----------------创建相关数据集----------------
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

print("x")
print(x)
print("y")
print(y)


plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

#----------------定义相关网络-----------------

# 首先，定义所有层属性
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出
    
    def forward(self, x):
        x = F.relu(self.hidden(x))      # 定义激活函数是relu 
        x = self.predict(x)             
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

#将相关网络结构打印出来
print(net)  

#----------------定义优化方法&定义损失函数-----------------
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

plt.ion()   # something about plotting

#----------------具体训练过程-----------------
for t in range(200):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()