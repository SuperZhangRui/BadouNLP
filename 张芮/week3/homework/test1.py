import numpy as np
import torch as torch
import torch.nn as nn
import random
import json
# 作业： 生成一组包含a的字符串， a在第几位就是第几类 如果扫描的长度离没有 a 则是第6类
# 创建模型
class TorchModel(nn.Module):
    # vector_num  字符集向量的维度      word_length  一段话最长多长    vocab 词库表
    def __init__(self,vector_num,word_length,vocab_size):
        super(TorchModel, self).__init__()
        # 1. embddding 层
        self.embedding = nn.Embedding(vocab_size, vector_num, padding_idx=0)
        # 池化成
        self.pool = nn.AvgPool1d(word_length)
        # 2. 线性层
        self.layer1 = nn.Linear(vector_num, word_length)
        ## 损失函数
        self.loss = nn.functional.cross_entropy
    def forward(self,x,y=None):
        x = self.embedding(x)
        x= x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.layer1(x)
        if y is not None:
            return self.loss(x,y)
        else:
            return x
# 生成词库表
def build_vocab():
    ku = "abcdfeghigklmnopqrstuvwxyz012"
    vocab = {'null':0} # 站空位置的
    for index,ci in enumerate(ku):
        vocab[ci] = index
    vocab['unk'] = len(vocab)  # 26
    return vocab
# 生成训练集
# 根据词汇库和单词长度生成单个的测试集
def build_sample(vocab,word_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(word_length-1)]
    x.append('a')
    random.shuffle(x)
    y = x.index('a')

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, word_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, word_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 用来测试模型
def evaluate(model, vocab, sample_length,word_length):
    test_x,test_y = build_dataset(sample_length, vocab, word_length)
    succ_count = 0 # 成功个数
    model.eval() # 测试模式
    with torch.no_grad(): # 不计算梯度
        pre_y = model(test_x)
        for pre_y1,true in zip(pre_y,test_y):
            pre_y2 = np.argmax(pre_y1)
            if(pre_y2 == true):
                succ_count += 1
    print(f"本次测试集数量{sample_length}, 正确率{succ_count/sample_length}")
    return succ_count/sample_length

# 开始训练模型

def mian():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    word_length = 6  # 样本文本长度
    learning_rate = 0.01  # 学习率

    vocab = build_vocab() # 构建词库
    model = TorchModel(char_dim,word_length,len(vocab))
    tran_x, tran_y=build_dataset(train_sample, vocab, word_length)
    ## 构建优化器
    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()  # 开启训练模式
        watch_loss = []
        for i in range(train_sample//batch_size):
            current_x = tran_x[i*batch_size:(i+1)*batch_size]
            current_y = tran_y[i*batch_size:(i+1)*batch_size]
            # print(current_y.shape)
            loss = model(current_x,current_y)
            loss.backward()
            adam.step()
            adam.zero_grad()
            watch_loss.append(loss.item())
        print(f"第{epoch}轮， loss均值为 {np.mean(watch_loss)}")
        evaluate(model, vocab, 20,word_length)

    # 最后保留权重
    torch.save(model.state_dict(),'model.pkl')
    # 保存此表
    file = open('vocab.json','w',encoding='utf-8')
    file.write(json.dumps(vocab,ensure_ascii=False, indent=2))
    file.close()
    return

if __name__ == '__main__':
    mian()