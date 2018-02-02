"""
    TODO: pretrain function and model checkpoint
"""

import sys
import time
import datetime
import logging
import cPickle as pickle
import os

import numpy as np

import torch

# import cv2
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from helpers.datagenerator import DataGenerator, FakeDataGenerator

# from generator import GeneratorConvEncDec, GeneratorEncDec, GeneratorEncDecTeacherForcing, GeneratorVan
# from generator import GeneratorEncDecTeacherForcingNoAtt as GeneratorEncDecTeacherForcing
from generator import GeneratorEncDecTeacherForcingNoAtt as GeneratorEncDecTeacherForcing
from discriminator import Discriminator

from helpers.utils import llprint
from helpers.evaluate import BleuScore

from loss import batchNLLLoss, JSDLoss#, MMDLoss


# candidates = ['This is a test','This is a test','This is a test','This is a test','This is a test','This is a test','This is a test']

# motions = ['This house would embrace multiculturalism',
# 'This house would ban gambling',
# 'This house would ban boxing',
# 'This house believes in the use of affirmative action',
# 'This house would permit the use of performance enhancing drugs in professional sports',
# 'This house supports the one-child policy of the republic of China',
# 'This house would build the Keystone XL pipeline']

generated = 'generated_latest.txt'
# generated = 'aaaa.txt'

candidates = []
motions = []

with open(generated) as fp:  
    line = fp.readline()
    while line:
        if '-----' in line or '=' in line:
            # print line.strip()
            line = fp.readline()
            continue
        elif 'believes that abortions should be legal' in line.lower():
            line = fp.readline()
            line = fp.readline()
            line = fp.readline()
            continue
        else:
            if '<' in line : 
                lineTemp = line.replace('< ','')
                lineTemp = lineTemp.replace('4','<UNK>')
                print ('Claim : %s'%lineTemp.strip())
                candidates.append(lineTemp.strip())
                print
            else : 
                lineTemp = line.replace('> ','')
                print ('Motion : %s'%lineTemp.strip())
                motions.append(lineTemp.strip())

            line = fp.readline()

if len(candidates) != len(motions): print 'Length is not the same'
else : BleuScore(candidate=candidates, motion=motions)