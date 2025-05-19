import matplotlib.pyplot as pyplot

## 准备训练集
X = [0.01 * x for x in range(100)]
Y = [2*x**2 + 4*x + 1 for x in X]

## 模型
def model(x):
    y = w1 * x**2 + w2 * x + w3
    return y

## 求 预测值与真实值对应的损失函数
def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2

## 给 权重 随机值
w1 ,w2, w3 = 1, 0, -1

batch_size = 10

## 开始机器学习
'''
整体思路:
 1. 首先生成测试集
 2. 定义需要学习的模型
 3. 定义损失函数
 4. 给权重添加随机值
 5. 定义批次 batch_count = 10  学习几组数据后更新一下权重值
 6. 还需要定义优化器， 即反向传播， 更新权重值
 7. 每次整个跑完一次测试集， 都统计当前学习到的权重和 本次的损失值的均值
 8. 当损失值达到 预期（0.0001）时停止学习
'''
lr = 0.1

for epoch in range(1000):
    epoch_loss = 0
    grad_w1 = 0
    grad_w2 = 0
    grad_w3 = 0
    counter = 0
    for x,y_true in zip(X,Y):
        y_prev = model(x) ## 计算预测值
        epoch_loss += loss(y_prev,y_true)   ## 计算预测值与真实值的差距 即：loss 将一组中的loss 加在一起 ，最后求平均， 可以直观地反应本轮 的loss
        counter += 1
        grad_w1 += 2 *  (y_prev - y_true) * x**2
        grad_w2 += 2 * (y_prev - y_true) * x
        grad_w3 += 2 * (y_prev - y_true)
        if counter == batch_size:
            # 权重更新
            w1 = w1 - lr * grad_w1 / batch_size  # sgd
            w2 = w2 - lr * grad_w2 / batch_size
            w3 = w3 - lr * grad_w3 / batch_size
            counter = 0
            grad_w1 = 0
            grad_w2 = 0
            grad_w3 = 0
    ## 此处测试集 跑了一遍了 统计本次的权重 以及 loss值
    epoch_loss =  epoch_loss/len(X)
    print("第%d轮，此时权重 w1:%f 、w2:%f 、w3:%f , loss %f" %(epoch,w1,w2,w3,epoch_loss))
    if epoch_loss < 0.00001:
        break


print(f"训练结束后：w1:{w1} w2:{w2} w3:{w3}")

# 使用训练后的模型生成一组预测集
Y1 = [ model(x)  for x in X]


#预测值与真实值比对数据分布
pyplot.scatter(X, Y, color="red")
pyplot.scatter(X, Y1)
pyplot.show()