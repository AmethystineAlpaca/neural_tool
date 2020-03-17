import torch
import numpy as np


class GCNPreprocessor:
    def __init__(self, docs):
        ori_docs = docs
        self.docs = {i: [set(doc), ori_doc]
                     for i, (doc, ori_doc) in enumerate(zip(docs, ori_docs))}
        self.tokens = {x for doc in docs for x in doc}
        self.vocabs = {i: x for i, x in enumerate(
            list(self.tokens)+list(self.docs.keys()))}


    def preprocess(self):

        adjacent_matrix = self.get_adjacent_matrix()
        degree_matrix = self.get_degree_matrix()
        docs_offset = len(self.tokens)
        return adjacent_matrix, degree_matrix, docs_offset

    def get_adjacent_matrix(self):
        adjacent_matrix = torch.zeros([len(self.vocabs), len(self.vocabs)])
        for i in range(len(adjacent_matrix)):
            for j in range(len(self.tokens)):
                if i == j:
                    adjacent_matrix[i][j] = 1
                elif i < len(self.tokens):
                    adjacent_matrix[i][j] = self.get_pmi(
                        self.vocabs[i], self.vocabs[j])
                else:
                    tfidf = self.get_tf_idf(self.vocabs[j], self.vocabs[i])
                    adjacent_matrix[i][j] = tfidf
        self.adjacent_matrix = adjacent_matrix
        return adjacent_matrix

    def get_degree_matrix(self):
        degree_matrix = torch.zeros(*self.adjacent_matrix.size())
        for i, row in enumerate(self.adjacent_matrix):
            degree_matrix[i][i] = torch.sum(row)
        return degree_matrix

    def get_tf_idf(self, token, doc_id):
        _, ori_doc = self.docs[doc_id]
        tf = ori_doc.count(token)/len(ori_doc)
        idf = len(self.docs)/len([i for i in self.docs if token in self.docs[i][0]])
        idf = np.log(idf)
        return tf*idf

    def get_pmi(self, token1, token2):
        co = len([i for i in self.docs if token1 in self.docs[i][0]
                  and token2 in self.docs[i][0]])
        # print(token1, token2)
        # print(self.docs)
        o_1 = len([i for i in self.docs if token1 in self.docs[i][0]])
        o_2 = len([i for i in self.docs if token2 in self.docs[i][0]])
        pmi = co/o_1*o_2

        return np.log(pmi) if pmi>0 else 0
