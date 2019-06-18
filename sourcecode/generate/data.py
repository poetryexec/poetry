# coding: UTF-8

import tensorflow as tf
import numpy as np
import argparse
import os
import random
import time
import collections
from generate import config as cf
class POEMS:
    "poem class"
    def __init__(self, config = cf.config(), isEvaluate=False):
        self.config = config
        """pretreatment"""
        poems = []
        file = open(self.config.trainPoems, "r")
        for line in file:  #every line is a poem
            title, author, poem = line.strip().split("::")  #get title and poem
            poem = poem.replace(' ','')
            if len(poem) < 10 or len(poem) > 512:  #filter poem
                continue
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                continue
            poem = '[' + poem + ']' #add start and end signs
            poems.append(poem)
            #print(title, author, poem)

        #counting words
        wordFreq = collections.Counter()
        for poem in poems:
            wordFreq.update(poem)
        # print(wordFreq)

        # erase words which are not common
        #--------------------bug-------------------------
        # word num less than original num, which causes nan value in loss function
        # erase = []
        # for key in wordFreq:
        #     if wordFreq[key] < 2:
        #         erase.append(key)
        # for key in erase:
        #     del wordFreq[key]

        wordFreq[" "] = -1
        wordPairs = sorted(wordFreq.items(), key = lambda x: -x[1])
        self.words, freq = zip(*wordPairs)
        self.wordNum = len(self.words)

        self.wordToID = dict(zip(self.words, range(self.wordNum))) #word to ID
        poemsVector = [([self.wordToID[word] for word in poem]) for poem in poems] # poem to vector
        if isEvaluate: #evaluating need divide dataset into test set and train set
            self.trainVector = poemsVector[:int(len(poemsVector) * self.config.trainRatio)]
            self.testVector = poemsVector[int(len(poemsVector) * self.config.trainRatio):]
        else:
            self.trainVector = poemsVector
            self.testVector = []
        print("训练样本总数： %d" % len(self.trainVector))
        print("测试样本总数： %d" % len(self.testVector))


    def generateBatch(self, isTrain=True):
        #padding length to batchMaxLength
        if isTrain:
            poemsVector = self.trainVector
        else:
            poemsVector = self.testVector

        random.shuffle(poemsVector)
        batchNum = (len(poemsVector) - 1) // self.config.batchSize
        X = []
        Y = []
        #create batch
        for i in range(batchNum):
            batch = poemsVector[i * self.config.batchSize: (i + 1) * self.config.batchSize]
            maxLength = max([len(vector) for vector in batch])
            temp = np.full((self.config.batchSize, maxLength), self.wordToID[" "], np.int32) # padding space
            for j in range(self.config.batchSize):
                temp[j, :len(batch[j])] = batch[j]
            X.append(temp)
            temp2 = np.copy(temp) #copy!!!!!!
            temp2[:, :-1] = temp[:, 1:]
            Y.append(temp2)
        return X, Y

