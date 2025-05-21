#  实现一个自己的线性层  和 pytorch的线性层作比较
import torch
import torch.nn as nn
import numpy as np

class TorchModel(nn.Module):
    def __init__(self,input_size,hidden_size,hidden_size2):
        super(TorchModel, self).__init__()
        ## 定义模型使用了哪些层
        self.layer1 = nn.Linear(input_size,hidden_size) # 输入 3，5
        self.layer2 = nn.Linear(hidden_size,hidden_size2)

    def forward(self,x):
        ## 根据定义的层计算值
        x = self.layer1(x)
        x = self.layer2(x)
        return x
## 随便定义一个测试数据   batch_size,
x = np.array([[3.1, 1.3, 1.2],
              [2.1, 1.3, 13]])
torchModel = TorchModel(3,5,2)
floatTensor = torch.FloatTensor(x)
y_pred = torchModel.forward(floatTensor)
print("预测值",y_pred)