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

from generator import GeneratorEncDec, GeneratorVan
from discriminator import Discriminator

from helpers.utils import llprint

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    gpu = 0

def cov(x):
    mean_x = torch.mean(x, 0)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    return c, mean_x

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class batchNLLLoss(nn.Module):
    def __init__(self):
        super(batchNLLLoss, self).__init__()

    def forward(self, synt, target, claim_length=20):
        """
        Returns the NLL Loss for predicting target sequence.
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()

        loss = 0

        synt = synt.permute(1,0,2)
        # _, indices = target.max(2)
        target = target.permute(1,0)
        for i in range(claim_length):
            # print synt[i]#.topk(1)[1]
            # print target[1]
            loss += loss_fn(synt[i], target[i])
            # print loss

        return loss

class batchNLLLossV2(nn.Module):
    def __init__(self):
        super(batchNLLLossV2, self).__init__()

    def forward(self, synt, target, claim_length=20):
        """
        Returns the NLL Loss for predicting target sequence.
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()

        loss = 0

        for i in range(synt.shape[0]):
            for j in range(claim_length):
                loss += loss_fn(synt[i][j].unsqueeze(0), target[i][j])

        return loss

class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss,self).__init__()

    def forward(self, f_real, f_synt):
        assert f_real.size()[1] == f_synt.size()[1]

        f_num_features = f_real.size()[1]
        batch_size = f_real.size()[0]
        identity = autograd.Variable(torch.eye(f_num_features)*0.1)

        if use_cuda:
            identity = identity.cuda(gpu)

        cov_real, mean_real = cov(f_real)
        cov_fake, mean_fake = cov(f_synt)

        f_real_mean = torch.mean(f_real, 0, keepdim=True)
        f_synt_mean = torch.mean(f_synt, 0, keepdim=True)

        dev_f_real = f_real - f_real_mean.expand(batch_size,f_num_features) # batch_size x num_feat
        dev_f_synt = f_synt - f_synt_mean.expand(batch_size,f_num_features) # batch_size x num_feat

        f_real_xx = torch.mm(torch.t(dev_f_real), dev_f_real) # num_feat x batch_size * batch_size x num_feat = num_feat x num_feat
        f_synt_xx = torch.mm(torch.t(dev_f_synt), dev_f_synt) # num_feat x batch_size * batch_size x num_feat = num_feat x num_feat

        cov_mat_f_real = f_real_xx / (batch_size) - torch.mm(f_real_mean, torch.t(f_real_mean)) + identity # num_feat x num_feat
        cov_mat_f_synt = f_synt_xx / (batch_size) - torch.mm(f_synt_mean, torch.t(f_synt_mean)) + identity # num_feat x num_feat

        # assert mean_real == f_real_mean.squeeze()
        # assert mean_fake == f_synt_mean.squeeze()
        assert cov_real == cov_mat_f_real
        assert cov_fake == cov_mat_f_synt

        cov_mat_f_real_inv = torch.inverse(cov_mat_f_real)
        cov_mat_f_synt_inv = torch.inverse(cov_mat_f_synt)

        temp1 = torch.trace(torch.add(torch.mm(cov_mat_f_synt_inv, torch.t(cov_mat_f_real)), torch.mm(cov_mat_f_real_inv, torch.t(cov_mat_f_synt))))
        temp1 = temp1.view(1,1)
        temp2 = torch.mm(torch.mm((f_synt_mean - f_real_mean), (cov_mat_f_synt_inv + cov_mat_f_real_inv)), torch.t(f_synt_mean - f_real_mean))
        loss_g = temp1 + temp2

        return loss_g

# class JSDLoss(nn.Module):
#     def __init__(self):
#         super(JSDLoss,self).__init__()
#
#     def forward(self, f_real, f_synt):
#         f_synt_target = torch.distributions.Normal.log_prob(autograd.Variable(f_synt.data))
#         f_real_target = torch.distributions.Normal.log_prob(autograd.Variable(f_real.data))
#
#         f_synt = torch.distributions.Normal.log_prob(f_synt)
#         f_real = torch.distributions.Normal.log_prob(f_real)
#
#         loss_g = (nn.KLDivLoss()(f_synt, f_synt_target) + nn.KLDivLoss()(f_synt, f_real_target) + nn.KLDivLoss()(f_real, f_real_target) + nn.KLDivLoss()(f_real, f_synt_target)) / 2
#         # sqrt_loss_g = torch.sqrt(loss_g)
#         print(loss_g)
#
#         return loss_g

class MMDCovLoss(nn.Module):
    def __init__(self):
        super(MMDCovLoss,self).__init__()

    def forward(self, batch_size, f_real, f_synt):
        """
            input: f_real , f_synt
                those are the extracted features of real claims and synthetic claims generated by the generator.
                size: batch_size x feature_dim
            output: loss_g
        """
        assert f_real.size()[1] == f_synt.size()[1]

        f_num_features = f_real.size()[1]
        identity = autograd.Variable(torch.eye(f_num_features)*0.1, requires_grad=False)

        if use_cuda:
            identity = identity.cuda(gpu)

        f_real_mean = torch.mean(f_real, 0, keepdim=True) #1 * num_features
        f_synt_mean = torch.mean(f_synt, 0, keepdim=True) #1 * num_features

        dev_f_real = f_real - f_real_mean.expand(batch_size,f_num_features) #batch_size * num_features
        dev_f_synt = f_synt - f_synt_mean.expand(batch_size,f_num_features) #batch_size * num_features

        f_real_xx = torch.mm(torch.t(dev_f_real), dev_f_real) #num_features * num_features
        f_synt_xx = torch.mm(torch.t(dev_f_synt), dev_f_synt) #num_features * num_features

        cov_mat_f_real = (f_real_xx / batch_size) - torch.mm(f_real_mean, torch.t(f_real_mean)) + identity #num_features * num_features
        cov_mat_f_synt = (f_synt_xx / batch_size) - torch.mm(f_synt_mean, torch.t(f_synt_mean)) + identity #num_features * num_features

        kxx, kxy, kyy = 0, 0, 0

        cov_sum = (cov_mat_f_fake + cov_mat_f_real)/2
        cov_sum_inv = torch.inverse(cov_sum)

        dividend = 1
        dist_x, dist_y = f_synt/dividend, f_real/dividend
        cov_inv_mat = cov_sum_inv
        x_sq = torch.sum(torch.mm(dist_x, cov_inv_mat) * dist_x, dim=1)
        y_sq = torch.sum(torch.mm(dist_y, cov_inv_mat) * dist_y, dim=1)

        tempxx = -2*torch.mm(torch.mm(dist_x, cov_inv_mat), torch.t(dist_x)) + x_sq + torch.t(x_sq)  # (xi -xj)**2
        tempxy = -2*torch.mm(torch.mm(dist_x, cov_inv_mat), torch.t(dist_y)) + x_sq + torch.t(y_sq)  # (xi -yj)**2
        tempyy = -2*torch.mm(torch.mm(dist_y, cov_inv_mat), torch.t(dist_y)) + y_sq + torch.t(y_sq)  # (yi -yj)**2

        for sigma in options['sigma_range']:
            kxx += torch.mean(torch.exp(-tempxx/2/(sigma**2)))
            kxy += torch.mean(torch.exp(-tempxy/2/(sigma**2)))
            kyy += torch.mean(torch.exp(-tempyy/2/(sigma**2)))
        loss_g = torch.sqrt(kxx + kyy - 2*kxy)

        return loss_g

class MMDLDLoss(nn.Module):
    def __init__(self):
        super(MMDLDLoss,self).__init__()

    def forward(self, batch_size, f_real, f_synt):
        """
            input: f_real , f_synt
                those are the extracted features of real claims and synthetic claims generated by the generator.
                size: batch_size x feature_dim
            output: loss_g
        """
        assert f_real.size()[1] == f_synt.size()[1]

        f_num_features = f_real.size()[1]
        identity = autograd.Variable(torch.eye(f_num_features)*0.1, requires_grad=False)

        if use_cuda:
            identity = identity.cuda(gpu)

        kxx, kxy, kyy = 0, 0, 0
        dividend = 32
        dist_x, dist_y = f_synt/dividend, f_real/dividend
        x_sq = torch.sum(dist_x**2, dim=1, keepdim=True)
        y_sq = torch.sum(dist_y**2, dim=1, keepdim=True)

        tempxx = -2*torch.mm(dist_x, torch.t(dist_x)) + x_sq + torch.t(x_sq)  # (xi -xj)**2
        tempxy = -2*torch.mm(dist_x, torch.t(dist_y)) + x_sq + torch.t(y_sq)  # (xi -yj)**2
        tempyy = -2*torch.mm(dist_y, torch.t(dist_y)) + y_sq + torch.t(y_sq)  # (yi -yj)**2

        for sigma in [20]:
            kxx += torch.sum(torch.exp(-tempxx/2/(sigma)))
            kxy += torch.sum(torch.exp(-tempxy/2/(sigma)))
            kyy += torch.sum(torch.exp(-tempyy/2/(sigma)))
        loss_g = torch.sqrt(kxx + kyy - 2*kxy)

        return loss_g
