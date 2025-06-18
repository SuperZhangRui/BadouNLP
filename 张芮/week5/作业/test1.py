from collections import defaultdict

import jieba
from gensim.models import Word2Vec
import numpy as np
import math
from sklearn.cluster import KMeans


# 模型训练  这个步骤忽略已经训练好了
def modeltrain(corpus, dim):
    model = Word2Vec(corpus, Vector_size=dim, sg=1)
    model.save('word2vec.w2v')
    # Word2Vec.load() 可以将训练好的词向量加载进来
    return model


def load_seance(path:str):
    # jieba分词
    seances = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            seances.add(" ".join(jieba.lcut(line)))
    print("共获得%d个句子" % len(seances))
    return seances

def sentences_to_vectors(sentences,model):
    vectors = []# 句向量
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                 vector += model.wv[word]
            except KeyError:
                vector +=np.zeros(model.vector_size)
        vectors.append(vector / len(words))# 对于词向量的和求平均是句向量
    return vectors
# 主进程： 训练词向量， 计算句向量， 合理选择质点个数,
def main():
    model = Word2Vec.load("model.w2v")
    seances = load_seance("titles.txt")
    vectors = sentences_to_vectors(seances,model) # 这一步获得了所有的句向量
    n_clusters = int(math.sqrt(len(seances)))  # 计算一个中心点的个数 用总句数
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    sentence_label_dict1 = defaultdict(list)
    for vector,seance,label  in zip(vectors,seances,kmeans.labels_):
        sentence_label_dict[label].append(seance)
        sentence_label_dict1[label].append(vector)

    intra_distances=[]
    for i in range(kmeans.n_clusters):
        vectors = sentence_label_dict1[i]
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        avg_distance = np.mean(distances)
        intra_distances.append(avg_distance)

    sorted_indices = np.argsort(intra_distances)[::-1]
    count = 0
    for a  in sorted_indices:
        count +=1
        str1 = sentence_label_dict[a]
        print("cluster %d:" % a)
        for str2 in str1:
            print(str2.replace(" ",""))
        if count == 10 :
            break


if __name__ == '__main__':
    main()
