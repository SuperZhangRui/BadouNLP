# 使用 dict1 分割 corpus 的文字斌不敢输出出来
import copy
import time
from ctypes import windll


#整理词库
def load_word_dict(path):
    maxlength = 0
    words = {}
    with open(path,'r',encoding='utf-8') as f:
        for line in f: #循环遍历每一行
            word = line.split()[0]
            words[word] = 1
            maxlength = max(maxlength, len(word))

    return words, maxlength

# 根据词组和最大词长切分字符串
def cut_linestr(maxlength,words,string):
    cutResult = []
    while string != "":
        string = string.strip()
        lens = min(maxlength,len(string)) # 当行的长度小于最大单词长的时候就直接使用句长
        word = string[:lens]
        while word not in words:
            if len(word) == 1:
                break
            word = string[:len(word)-1]
        cutResult.append(word)
        string = string[len(word):]
    return cutResult
##########################################  上边是第一种方法  ##############################################################
# 第二种的 分词方法
'''
    思路： 基于第一种分词方法， 将每个词表中存在的词，都继续从左细分 1个字、2个字、3个字  在填充到词表中 不过词典对应的值为 0， 0表示在词表中不存在
'''
def load_word_dict2(path):
    words,maxlength = load_word_dict(path)
    print("第一种分词方法，词库长度 %d" % len(words))
    returnwords = copy.deepcopy(words)
    for word in words.keys():
        for i in range(len(word)):
            if i+1 == len(word):
                break
            str1 = word[:i+1]
            if str1 not in returnwords:
                returnwords[str1] = 0
    print("第二种分词方法，词库长度 %d" % len(returnwords))
    return returnwords,maxlength
def cut_linestr2(words,string):
    curResult = []
    if string == "":
        return curResult
    start_index,end_index = 0,1 # 两个索引框住一个窗口
    window = string[start_index:end_index]
    find_word = window
    # 记录当前可以计入结果的词,如目前的窗口是一个词 或者目前的单字
    while start_index < len(string):
        if window not in words or end_index > len(string):
            curResult.append(find_word)
            start_index += len(find_word) #更新起点
            end_index += 1
            window = string[start_index:end_index]
            find_word = window
        #如果是一个前缀
        elif words[window] == 0:
            end_index += 1
            window = string[start_index:end_index]

        # 如果是一个词
        elif words[window] == 1:
            find_word = window
            end_index += 1
            window = string[start_index:end_index]
        # 最后找到的window如果不在词典里，把单独的字加入切词结果
    if words.get(window) != 1:
        curResult += list(window)
    else:
        curResult.append(window)
    return curResult


def main(words ,maxLength,inputputh,out_filePath):
    start_time = time.time()
    with open(out_filePath,'w',encoding='utf-8') as w:
        with open(inputputh,'r',encoding='utf-8') as r:
            for line in r:
                w.write(" / ".join(cut_linestr(maxLength,words,line))+"\n")
                #w.write(" / ".join(cut_linestr2(words,line))+"\n")
    print("耗时：", time.time() - start_time)
    return


def main2(words ,maxLength,inputputh,out_filePath):
    start_time = time.time()
    with open(out_filePath,'w',encoding='utf-8') as w:
        with open(inputputh,'r',encoding='utf-8') as r:
            for line in r:
                #w.write(" / ".join(cut_linestr(maxLength,words,line))+"\n")
                w.write(" / ".join(cut_linestr2(words,line)))
    print("耗时：", time.time() - start_time)
    return



words, maxLength = load_word_dict("dict.txt")
words2, maxLength1 = load_word_dict2("dict.txt")

if __name__ == '__main__':
    main(words ,maxLength,"corpus.txt", "cut_method1_output.txt")
    main2(words2 ,maxLength,"corpus.txt", "cut_method2_output.txt")
