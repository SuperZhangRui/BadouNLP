# 就按课上的代码 训练一个模型吧
import torch
import torch.nn as  nn
import numpy as np
import matplotlib.pyplot as plt

# 创建自己的模型
class MyModel(nn.Module):
    def __init__(self,input_size):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 1)
        self.avtivation = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, input,y_true=None):
        x = self.layer1(input)
        x = self.avtivation(x)
        if y_true is not None:
            return self.loss(x,y_true) ## 训练模式计算损失值
        else:
            return x  ## 非训练模式 直接计算预测值

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate(model):
    model.eval() # 开启训练模式
    test_sample_num = 100 # 用于测试的数据量
    test_X,test_Y = build_dataset(test_sample_num)
    successcount = 0
    with torch.no_grad():
        for x,y_true in zip(test_X, test_Y):
            y_prev = model(x)
            if y_prev >= 0.5 and y_true == 1:
                successcount += 1
            elif y_prev < 0.5 and y_true == 0:
                successcount += 1
    bili = successcount / test_sample_num
    print(f"本轮训练结束,模型的正确率{bili:.3f}")
    return bili

def main():
    # 准备训练
    bath_size = 20 # 一批数据多少
    test_sample_num = 5000 # 训练集的数据量
    epoch_num = 20 # 共训练多少论
    lr = 0.001 # 步长 # 忘了另一个名字了
    input_size = 5 # 忘了叫啥了
    model = MyModel(input_size)

    # train data list 训练集
    train_X,train_Y = build_dataset(test_sample_num)
    ## 初始化优化器
    optim =torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for epoch in range(epoch_num):
        model.train() # 进入训练模式
        watchLoss = []
        for i in range(test_sample_num // bath_size):
            x = train_X[i*bath_size:(i+1)*bath_size]
            y_true = train_Y[i*bath_size:(i+1)*bath_size]
            loss = model(x,y_true) # 获取损失函数
            loss.backward() ## 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() #梯度归零
            watchLoss.append(loss.item())
        print(f"第{epoch}轮 , 平均损失值为{np.mean(watchLoss):.5f}") ## 试试这个语法
        # 每轮训练完都测试一下这一轮的成功率
        acc = evaluate(model)
        log.append([acc, float(np.mean(watchLoss))])
    ## 训练结束了， 保存一下模型的权重吧
    torch.save(model.state_dict(),'model.ini')
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
