#week3作业
from collections import defaultdict
import copy
#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO 具体思路 lenList是 key为长度  value为 list(list) 的字典
    lenList = defaultdict(list)
    for i in range(1,len(sentence)+1):
        word = sentence[0:i]
        for dict_word in Dict:
            if len(dict_word) <= i and word[-len(dict_word):] == dict_word:
                if(len(lenList) == 0):
                    lenList[1] = [[word]]
                else:
                    a = lenList[i-len(dict_word)]
                    if(a == []):
                        lenList[i] = [[dict_word]]
                    else:
                        for b in a:
                            b  = copy.deepcopy(b)
                            b.append(dict_word)
                            lenList[i].append(b)
    return lenList[7]


  #  return target

if __name__ == "__main__":
    targetList = all_cut(sentence, Dict)
    for a in targetList:
        print(a)
#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

