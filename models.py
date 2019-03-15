"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import math


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.2, device=None):
        super(BiDAF, self).__init__()
        hidden_size = 96
        self.hidden_size = hidden_size
        self.device = device
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=96,
                                    drop_prob=drop_prob)

        # self.enc = layers.RNNEncoder(input_size=hidden_size,
        #                              hidden_size=hidden_size,
        #                              num_layers=1,
        #                              drop_prob=drop_prob)

        self.emb_encoder = layers.EmbeddingEncoder()

        self.att = layers.BiDAFAttention(hidden_size=96,
                                         drop_prob=drop_prob)

        # self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
        #                              hidden_size=hidden_size,
        #                              num_layers=2,
        #                              drop_prob=drop_prob)

        self.model_encoder = layers.ModelEncoder(d_model=hidden_size, drop_prob=drop_prob, device=device)

        # self.out = layers.BiDAFOutput(hidden_size=hidden_size,
        #                               drop_prob=drop_prob)
        #

        self.out = layers.QANet(hidden_size=hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # print('c_mask: ', c_mask.shape)


        c_emb = self.emb(cw_idxs, cc_idxs).to(self.device)       # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs).to(self.device)         # (batch_size, q_len, hidden_size)

        c_emb_out = self.emb_encoder(c_emb, c_mask).to(self.device)
        q_emb_out = self.emb_encoder(q_emb, q_mask).to(self.device)
        # print('c_emb_out: ', c_emb.shape)
        # print('q_emb_out: ', q_emb.shape)

        # c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        # q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_emb_out, q_emb_out,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        att = att.permute(0, 2, 1)

        conv = nn.Conv1d(self.hidden_size * 4, self.hidden_size, 7, padding=math.floor(7/2))
        att = conv(att)

        att = att.permute(0, 2, 1)

        # print(att.shape)

        M1, M2, M3 = self.model_encoder(att, c_mask)

        # print('shapes over herreeee')
        #
        # print(M1.shape)
        # print(M2.shape)
        # print(M3.shape)

        # mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(M1, M2, M3, c_mask)  # 2 tensors, each (batch_size, c_len)

        # out = self.out(att, mod, c_mask)

        return out
