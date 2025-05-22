'''
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
'''
import torch
from torch import nn
import numpy as np

# 自己定义一个变量用于记录我本次的训练的数据
input_size = 5

# 生成测试数据
# 模型的训练数据是 五维向量计算分类
def generator():
    x = np.random.random(input_size)
    # 找到列表x中最大的数 以及它的索引
    return x,np.argmax(x)

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x ,y = generator()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(X), torch.LongTensor(Y)

# 构建自己的神经网络
class TensorModel(nn.Module):
    def __init__(self,input_size): # 只需要指定输入的维度 因为输出的维度是5
        super(TensorModel, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size)
        #self.act = nn.Sigmoid()
    def forward(self, input,y_true=None): # 用于计算预测值
        x = self.layer1(input)
        if y_true is not None:
            return nn.CrossEntropyLoss()(x,y_true)
        else:
            return x
# 训练模型

def main():
    trainCount = 50 # 共训练20轮
    trainCount_pre = 20 # 每次训练的样本数
    total_sample_num = 5000 # 测试集的大小
    lr = 0.001 # 学习率
    # 创建模型对象
    model = TensorModel(input_size)
    tensor_X,tensor_Y = build_dataset(total_sample_num) # 生成训练集
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)


    for epoce in range(trainCount):
        # 进入每组数据的训练中
        model.train() # 模型进入训练模式
        watch_loss = [] # 用来统计本次的损失值之和
        for i in range(total_sample_num // trainCount_pre):
            x = tensor_X[i*trainCount_pre:(i+1)*trainCount_pre]
            y = tensor_Y[i*trainCount_pre:(i+1)*trainCount_pre]
            loss = model(x,y)
            loss.backward() # 求梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())

        print("第%d轮 ， 损失值为%f" %(epoce,np.mean(watch_loss)))
        evaluate(model)
    torch.save(model.state_dict(),"model.bin") # 保存本次


def evaluate(model):
    # 测试当前的模型
    total_sample_num = 100
    tensor_X, tensor_Y = build_dataset(total_sample_num)
    succNum = 0
    with torch.no_grad():
        logits = model(tensor_X)
        pred_labels = torch.argmax(logits,dim=1)
        for y_prev,y_true in zip(pred_labels, tensor_Y):
            if(y_prev == y_true):
                succNum += 1
    print("本次测试的正确率是%f" % (succNum / total_sample_num))

def predict(model_path): #测试
    model = TensorModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())
    evaluate(model)

if __name__ == '__main__':
    main()
    #predict("model.bin")