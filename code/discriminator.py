import sys
import time
import datetime
import logging
import cPickle

import numpy as np

import torch

# import cv2
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from helpers.datagenerator import DataGenerator

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    gpu = 0

class DiscriminatorEmb(nn.Module):

    def __init__(self, batch_size, vocab_size, claim_size, hidden_dim, n_layers=2, dropout_p=0.5):
        super(Discriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.conv_1 = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv_2 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv_3 = nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2)
        self.pool3 = nn.MaxPool1d(2)
        # self.conv_3 = nn.Conv1d(hidden_dim, hidden_dim, 5)
        # self.pool3 = nn.AvgPool1d(2)
        # self.gru_1 = nn.GRU(vocab_size, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(1500, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, claim):
        # hidden = autograd.Variable(torch.zeros(1, self.claim_length, self.hidden_dim))
        # if use_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden
        claim = self.embeddings(claim)
        claim = self.dropout(claim)
        claim = claim.permute(0,2,1)

        # x, hidden = self.gru_1(claim, None) #[batch_size, seq_length, hidden_dim]
        x_1 = F.relu(self.conv_1(claim))
        x_1 = self.dropout(x_1)
        x_1 = self.pool1(x_1)

        x_2 = F.relu(self.conv_2(claim))
        x_2 = self.dropout(x_2)
        x_2 = self.pool2(x_2)

        x_3 = F.relu(self.conv_3(claim))
        x_3 = self.dropout(x_3)
        x_3 = self.pool3(x_3)

        x = torch.cat((x_1, x_2, x_3), dim=-1)

        x = x.contiguous().view(self.batch_size, -1)
        # x = x.view(self.batch_size, -1)
	# print x
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        output = self.sigmoid(self.fc_out(x))

        return x, output

class Discriminator(nn.Module):

    def __init__(self, batch_size, vocab_size, claim_size, hidden_dim, n_layers=2, dropout_p=0.5):
        super(Discriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.conv_1 = nn.Conv1d(vocab_size, hidden_dim, 3, stride=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv_2 = nn.Conv1d(vocab_size, hidden_dim, 4, stride=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv_3 = nn.Conv1d(vocab_size, hidden_dim, 5, stride=2)
        self.pool3 = nn.MaxPool1d(2)
        # self.conv_3 = nn.Conv1d(hidden_dim, hidden_dim, 5)
        # self.pool3 = nn.AvgPool1d(2)
        # self.gru_1 = nn.GRU(vocab_size, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(3600, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, claim):
        # hidden = autograd.Variable(torch.zeros(1, self.claim_length, self.hidden_dim))
        # if use_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden
        claim = claim.permute(0,2,1)

        # x, hidden = self.gru_1(claim, None) #[batch_size, seq_length, hidden_dim]
        x_1 = F.relu(self.conv_1(claim))
        x_1 = self.dropout(x_1)
        x_1 = self.pool1(x_1)

        x_2 = F.relu(self.conv_2(claim))
        x_2 = self.dropout(x_2)
        x_2 = self.pool2(x_2)

        x_3 = F.relu(self.conv_3(claim))
        x_3 = self.dropout(x_3)
        x_3 = self.pool3(x_3)

        x = torch.cat((x_1, x_2, x_3), dim=-1)

        x = x.contiguous().view(self.batch_size, -1)
        # x = x.view(self.batch_size, -1)
	# print x
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        output = self.sigmoid(self.fc_out(x))

        return x, output

class DiscriminatorV2(nn.Module):

    def __init__(self, batch_size, vocab_size, claim_size, hidden_dim, n_layers=2, dropout_p=0.5):
        super(DiscriminatorV2, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.conv_1 = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv_2 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv_3 = nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2)
        self.pool3 = nn.MaxPool1d(2)
        # self.conv_3 = nn.Conv1d(hidden_dim, hidden_dim, 5)
        # self.pool3 = nn.AvgPool1d(2)
        # self.gru_1 = nn.GRU(vocab_size, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(3600, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_sent):
        # hidden = autograd.Variable(torch.zeros(1, self.input_sent_length, self.hidden_dim))
        # if use_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden
        if(len(input_sent.shape)==3):
            topv, topi = input_sent.data.topk(1)
            ni = topi
            # input_sent_v = autograd.Variable(torch.LongTensor([[ni]]))
            input_sent = autograd.Variable(torch.LongTensor(ni))
            
        input_sent = input_sent.cuda() if use_cuda else input_sent

        input_sent = self.embeddings(input_sent.squeeze())
        input_sent = self.dropout(input_sent)
        input_sent = input_sent.permute(0,2,1)

        # x, hidden = self.gru_1(input_sent, None) #[batch_size, seq_length, hidden_dim]
        x_1 = F.relu(self.conv_1(input_sent))
        x_1 = self.dropout(x_1)
        x_1 = self.pool1(x_1)

        x_2 = F.relu(self.conv_2(input_sent))
        x_2 = self.dropout(x_2)
        x_2 = self.pool2(x_2)

        x_3 = F.relu(self.conv_3(input_sent))
        x_3 = self.dropout(x_3)
        x_3 = self.pool3(x_3)

        x = torch.cat((x_1, x_2, x_3), dim=-1)

        x = x.contiguous().view(self.batch_size, -1)
        # x = x.view(self.batch_size, -1)
	    # print x
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        output = self.sigmoid(self.fc_out(x))

        return x, output
