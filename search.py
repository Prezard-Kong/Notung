from __future__ import unicode_literals
from eval import Evaluator
import util
from sklearn.neighbors import KDTree
import numpy as np
from nltk import word_tokenize, pos_tag
import os
import cPickle as pickle


weights = {'NOUN': 1.0, 'VERB': 1.0, 'ADJ': 0.5, 'ADV': 0.5, 'ADP': 0.1, 'CONJ': 0.1, 'DET': 0.1,
           'NUM': 0.1, 'PRT': 0.1, 'PRON': 0.1, 'X': 0, '.': 0}


def tags2map(tags):
    ans = {}
    for t in tags:
        ans[t[0]] = t[1]
    return ans


def calc_weighted_sums(keywords, query):

    tags = set(pos_tag(word_tokenize(query), tagset='universal'))
    words = tags2map(tags)

    def calc_weighted_sum(keyword):
        tmp = set(keyword)
        ans = 0.0
        for t in tmp:
            if t in words:
                ans += weights[words[t]]
        return ans

    ans = []
    for i, k in enumerate(keywords):
        ans.append((calc_weighted_sum(k), i))
    ans.sort(key=lambda key: key[0], reverse=True)
    return ans


class ImageSearcher(object):

    def __init__(self, database='./data/database/data.pickle'):
        if os.path.exists(database):
            with open(database) as f:
                self.__KDTree = pickle.load(f)
                self.__keywords = pickle.load(f)
                self.__paths = pickle.load(f)
        else:
            self.__KDTree, self.__keywords, self.__paths  = None, None, None
        self.__database = database
        self.__eval = Evaluator()

    def build_database(self, path):
        if self.__keywords is None:
            self.__keywords = []
        if self.__paths is None:
            self.__paths = []
        candidates = []
        for pa in path:
            candidates += [os.path.join(pa, p) for p in os.listdir(pa)] if os.path.isdir(pa) else [pa]
        tmp = np.empty((len(candidates), 25088), dtype=np.float32)  # 25088
        i = 0
        for c in candidates:
            img = util.load_img(c)
            if img is None:
                continue
            self.__eval.evaluate(img)
            tmp[i, :] = self.__eval.conv_features
            self.__keywords.append(word_tokenize(self.__eval.caption))
            self.__paths.append(c)
            i += 1
        if i > 0:
            tmp = tmp[:i, :]
            features = np.vstack((self.__KDTree.data, tmp)) if self.__KDTree is not None else tmp
            self.__KDTree = KDTree(features)
            if os.path.exists(self.__database):
                os.remove(self.__database)
            if not os.path.exists(os.path.dirname(self.__database)):
                os.mkdir(os.path.dirname(self.__database))
            with open(self.__database, 'a') as f:
                pickle.dump(self.__KDTree, f, protocol=2)
                pickle.dump(self.__keywords, f, protocol=2)
                pickle.dump(self.__paths, f, protocol=2)

    def image_similarity_search(self, img_path, top_k=10):
        img = util.load_img(img_path)
        if img is None or self.__KDTree is None:
            return None, None
        self.__eval.evaluate(img)
        index = self.__KDTree.query(self.__eval.conv_features, k=top_k, return_distance=False)
        paths = []
        captions = []
        for i in index[0]:
            paths.append(self.__paths[i])
            captions.append(' '.join(self.__keywords[i]))
        return paths, captions

    def semantic_similarity_search(self, query, top_k=10):
        if self.__keywords is None:
            return None, None
        sums = calc_weighted_sums(self.__keywords, query)
        paths = []
        captions = []
        for i in range(top_k):
            paths.append(self.__paths[sums[i][1]])
            captions.append(' '.join(self.__keywords[sums[i][1]]))
        return paths, captions



