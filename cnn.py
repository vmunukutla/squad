#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F


# class CNN(torch.nn.Module):
#     def __init__(self, e_word, e_char=50, m_word=21, kernel_size=5):
#
#         super(CNN, self).__init__()
#         self.kernel_size = kernel_size
#         self.e_char = e_char
#         self.e_word = e_word
#         self.m_word = m_word
#         self.conv = nn.Conv1d(e_char, e_word, kernel_size=kernel_size, stride=1, bias=True)
#         self.max_pool = nn.MaxPool1d(kernel_size=self.m_word-self.kernel_size+1)
#
#     def forward(self, x_reshaped):
#     	x_conv = self.conv(x_reshaped)
#     	x_convoutput = F.relu(x_conv)
#     	x_convoutput = self.max_pool(x_convoutput)
#     	return x_convoutput

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, e_word, e_char, kernel_size=5):
        super(CNN, self).__init__()

        self.conv_layer = nn.Conv1d(e_char, e_word, kernel_size)

    def forward(self, x_reshaped):
        x_conv = self.conv_layer(x_reshaped)
        x_conv_out = torch.max(F.relu(x_conv), 2)[0]
        return x_conv_out

class PointwiseCNN(nn.Module):

    def __init__(self, e_word, e_char, kernel_size=1):
        super(PointwiseCNN, self).__init__()
        self.conv_layer = nn.Conv1d(e_char, e_word, kernel_size)

    def forward(self, x_reshaped):
        x_conv = self.conv_layer(x_reshaped)
        return x_conv
