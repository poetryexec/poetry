# coding: UTF-8

import tensorflow as tf
import numpy as np
import argparse
import os
import random
import time
import collections

class config(object):
    def __init__(self,batchSize=64,learningRateBase=0.001,learningRateDecayStep=1000,learningRateDecayRate=0.95,epochNum=10,saveStep=1000,type='poetrySong',generateNum=1,trainRatio=0.8):
        self.root_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
        self.batchSize = batchSize

        self.learningRateBase = learningRateBase
        self.learningRateDecayStep = learningRateDecayStep
        self.learningRateDecayRate = learningRateDecayRate
        self.epochNum = epochNum  # train epoch
        self.type = type
        self.generateNum = generateNum
        self.trainPoems = self.root_dir + "dataset/" + self.type + "/" + self.type + ".txt" # training file location
        self.checkpointsPath = self.root_dir + "checkpoints/" + self.type  # checkpoints location
        self.saveStep = saveStep  # save model every savestep

        # evaluate
        self.trainRatio = trainRatio  # train percentage
        self.evaluateCheckpointsPath = self.root_dir + "checkpoints/evaluate"


