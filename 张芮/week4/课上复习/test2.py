#新词发现
import math
from collections import defaultdict

class NewWordDected:
    def __init__(self,path):
        self.word_count = defaultdict(int) # 记录每个单词的个数
        self.left_neighbor = defaultdict(dict) # 记录每个单词的个数
        self.right_neighbor = defaultdict(dict) # 记录每个单词的个数
        # 最多几grem
        self.maxWordLength = 5
        self.load_corpus(path)
        self.calc_pmi()
        self.calc_entropy()
        self.calc_word_values()

    # 加载语料数据，并进行统计
    def load_corpus(self, path):
        with open(path,'r',encoding='utf8') as f:
            for line in f:
                line = line.strip()
                for i in range(1,self.maxWordLength+1):
                    self.ngram_count(line, i)
        return

    # 按照窗口长度取词,并记录左邻右邻
    def ngram_count(self, line, word_length):
        for i in range(len(line)-word_length+1):
            word = line[i:i+word_length]
            self.word_count[word] += 1
            # 记录左邻个数和出现的次数
            if i -1 >= 0:
                char = line[i-1]
                self.left_neighbor[word][char] =  self.left_neighbor[word].get(char, 0) + 1
            #记录友邻个数和出现的次数
            if i +word_length < len(line):
                char = line[i +word_length]
                self.right_neighbor[word][char] =  self.right_neighbor[word].get(char, 0) + 1
        return

    def calc_pmi(self):# 计算每个词出现的概率
        self.word_occurrence_p = defaultdict(float)
        self.word_length_counts = defaultdict(int)
        # 首先记录每个长度的词出现的次数
        self.word_length_counts = defaultdict(int)
        for word in self.word_count.keys():
            self.word_length_counts[len(word)] += self.word_count[word]
        self.pmi = {}
        for word, count in self.word_count.items():
            p_word = count / self.word_length_counts[len(word)]
            p_chars = 1
            for char in word:
                p_chars *= self.word_count[char] / self.word_length_counts[1]
            self.pmi[word] = math.log(p_word / p_chars, 10) / len(word)
        return

        # 计算熵

    def calc_entropy_by_word_count_dict(self, word_count_dict):
        total = sum(word_count_dict.values())
        entropy = sum([-(c / total) * math.log((c / total), 10) for c in word_count_dict.values()])
        return entropy

        # 计算左右熵

    def calc_entropy(self):
        self.word_left_entropy = {}
        self.word_right_entropy = {}
        for word, count_dict in self.left_neighbor.items():
            self.word_left_entropy[word] = self.calc_entropy_by_word_count_dict(count_dict)
        for word, count_dict in self.right_neighbor.items():
            self.word_right_entropy[word] = self.calc_entropy_by_word_count_dict(count_dict)

    def calc_word_values(self):
        self.word_values = {}
        for word in self.pmi:
            if len(word) < 2 or "，" in word:
                continue
            pmi = self.pmi.get(word, 1e-3)
            le = self.word_left_entropy.get(word, 1e-3)
            re = self.word_right_entropy.get(word, 1e-3)
            self.word_values[word] = pmi * le * re

if __name__ == "__main__":
    nwd = NewWordDected("sample_corpus.txt")
    # print(nwd.word_count)
    # print(nwd.left_neighbor)
    # print(nwd.right_neighbor)
    # print(nwd.pmi)
    # print(nwd.word_left_entropy)
    # print(nwd.word_right_entropy)
    value_sort = sorted([(word, count) for word, count in nwd.word_values.items()], key=lambda x: x[1],
                        reverse=True)
    print([x for x, c in value_sort if len(x) == 2][:10])
    print([x for x, c in value_sort if len(x) == 3][:10])
    print([x for x, c in value_sort if len(x) == 4][:10])