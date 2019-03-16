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
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        hidden_size = 96
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.emb_encoder = layers.EmbeddingEncoder(drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.model_encoder = layers.ModelEncoder(d_model=hidden_size, drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)


        self.conv_layer = nn.Conv1d(hidden_size * 4, hidden_size, 7, padding=math.floor(7/2))
        self.out = layers.QANet(hidden_size=hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # print('c_mask: ', c_mask.shape)

        print('cw_idxs')
        print(cw_idxs)
        print('qw_idxs')
        print(qw_idxs)
        print('cc_idxs')
        print(cc_idxs)
        print('qc_idxs')
        print(qc_idxs)
        c_emb = self.emb(cw_idxs, cc_idxs)      # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)        # (batch_size, q_len, hidden_size)
        print('c_emb')
        print(c_emb)
        print(c_emb.shape)
        print('q_emb')
        print(q_emb)
        print(q_emb.shape)

        c_enc = self.emb_encoder(c_emb, c_mask) # (batch_size, c_len, hidden_size)
        q_enc = self.emb_encoder(q_emb, q_mask) # (batch_size, c_len, hidden_size)
        # print('c_enc')
        # print(c_enc)
        # print(c_enc.shape)
        # print('q_enc')
        # print(q_enc)
        # print(q_enc.shape)

        # c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        # q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)
        #
        # print('att')
        # print(att)
        # print(att.shape)
        att = att.permute(0, 2, 1)
        att = self.conv_layer(att)

        att = att.permute(0, 2, 1)

        M1, M2, M3 = self.model_encoder(att, c_mask)

        # mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(M1, M2, M3, c_mask)  # 2 tensors, each (batch_size, c_len)

        # out = self.out(att, mod, c_mask)

        return out
