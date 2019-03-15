"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from cnn import CNN, PointwiseCNN
import numpy as np


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.cnn = CNN(word_vectors.size(1), char_vectors.size(1))
        self.highway = Highway(word_vectors.size(1))
        self.proj = nn.Linear(word_vectors.size(1) * 2, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x, y):
        input_shape = y.shape # (batch_size, seq_len, 16) = (64, seq_len, 16)
        input_reshaped = torch.reshape(y, (input_shape[0] * input_shape[1], input_shape[2])) # (64*seq_len, 16)
        #print(input_reshaped.shape)
        x_padded = self.char_embed(input_reshaped) # 64-dimensional
        x_reshaped = x_padded.permute(0, 2, 1) # (64*seq_len, 64, 16)
        #print(x_reshaped.shape)
        #print('hello')
        x_conv_out = self.cnn.forward(x_reshaped) # (64*seq_len, 300)
        #print(x_conv_out.shape)

        x_highway = self.highway(x_conv_out)
        x_highway_reshaped = torch.reshape(x_highway, (input_shape[0], input_shape[1], x_highway.shape[1]))
        x_word_emb = F.dropout(x_highway_reshaped, self.drop_prob, self.training)

        #print(x_word_emb.shape)

        word_emb = self.word_embed(x)   # (batch_size, seq_len, embed_size)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        word_emb = torch.cat((word_emb, x_word_emb), dim=2)
        #print('first shape')
        word_emb = self.proj(word_emb)  # (batch_size, seq_len, hidden_size)
        word_emb = self.hwy(word_emb)   # (batch_size, seq_len, hidden_size)
        #print(word_emb.shape)

        return word_emb # (batch_size, seq_len, 2 * embed_size) = (64, seq_len, 100)

class Highway(nn.Module):

    def __init__(self, embed_size):
        super(Highway, self).__init__()

        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        x_proj = F.relu(self.projection(x))
        x_gate = F.sigmoid(self.gate(x))
        x_highway = torch.mul(x_proj, x_gate) + torch.mul(torch.add(1, torch.mul(x_gate, -1)), x)
        return x_highway


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

# borrowed from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 400):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        #add back .cuda() if necessary
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], \
        requires_grad=False)
        return x

#https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class FeedForwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # comment and uncomment to try out
        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out

# 1. Got embedding of shape (batch_size, sq_len, embed_size * 2) -> (64, sq_len, 600)

# (batch_size, sq_len, 128)
# 2. First layernorm is of shape [sq_len, embed_size * 2]
# 3. First conv maps [batch_size, sq_len, embed_size * 2] -> [batch_size, sq_len, hidden_size] (permute to do this)
# 4. add original with conv result's second dimension, shape -> [batch_size, sq_len, d_model + 600] -> [64, sq_len, d_model + 600]
# 5. Repeat. But if I do this, how is the final dimension d = 128 for the output if I keep concatenating?
# 6. What exactly is PointWise convolution? Also, my CNN currently takes the max, so it completely erases the last dimension
# 7. CQ Attention Layer is the same as the BiDAFAttention class
# 8. what does it mean to share weights in the 4th step?
# 9. what do the arrows mean?



class EmbeddingEncoder(nn.Module):

    def __init__(self, kernel_size=7, d_model=96, num_layers=4, drop_prob=0.2):
        super(EmbeddingEncoder, self).__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.conv_layers = [PointwiseCNN(d_model, d_model, kernel_size) for i in range(num_layers)]
        self.attention = MultiHeadAttention(d_model=d_model)
        self.layer_norm = [nn.LayerNorm(d_model) for i in range(num_layers+2)]
        self.feed_forward = FeedForwardNeuralNetModel(d_model, int(d_model/2), d_model)
        self.pos_encoder = PositionalEncoder(d_model=d_model)
        self.drop_prob = drop_prob

    def forward(self, input, mask):
        # print('start')
        prev_out = input
        print(prev_out.dtype)
        # print(prev_out.shape)
        prev_out = self.pos_encoder(prev_out)
        print(prev_out.dtype)
        # print(prev_out.shape)
        for i in range(self.num_layers):
            #print(prev_out.shape)
            layer_out = self.layer_norm[i](prev_out)
            layer_out = layer_out.permute(0, 2, 1)
            # print('layer shape')
            # print(layer_out.shape)
            conv_out = self.conv_layers[i].forward(layer_out)
            #print(conv_out.shape)
            conv_out = conv_out.permute(0, 2, 1)
            #print(conv_out.shape)
            concat_out = conv_out + prev_out
            concat_out = F.dropout(concat_out, p=self.drop_prob, training=self.training)
            #print(concat_out.shape)
            prev_out = concat_out
        layer_out = self.layer_norm[self.num_layers](prev_out)
        attention_out = self.attention(layer_out, mask)
        concat_out = prev_out + attention_out
        concat_out = F.dropout(concat_out, p=self.drop_prob, training=self.training)
        prev_out = concat_out
        # print('look here')
        # print(concat_out.shape)
        layer_out = self.layer_norm[self.num_layers+1](prev_out)
        #print(layer_out.shape)
        feed_out = self.feed_forward(layer_out)
        #print(feed_out.shape)
        concat_out = concat_out + feed_out
        concat_out = F.dropout(concat_out, p=self.drop_prob, training=self.training)
        #print(concat_out.shape)
        # print('here')
        # print(concat_out.shape)
        return concat_out


# borrowed from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # print('mask off')
            # print(attn.shape)
            mask = mask.view(mask.size(0), mask.size(1), 1)
            # print(mask.shape)
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=4, d_model=128, drop_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model
        self.d_v = d_model
        W_Q = [nn.Parameter(torch.zeros(d_model, self.d_k)) for i in range(num_heads)]
        W_K = [nn.Parameter(torch.zeros(d_model, self.d_k)) for i in range(num_heads)]
        W_V = [nn.Parameter(torch.zeros(d_model, self.d_v)) for i in range(num_heads)]
        self.W_O = nn.Parameter(torch.zeros(num_heads * self.d_v, d_model))
        nn.init.xavier_uniform_(self.W_O)
        for i in range(num_heads):
            for weight in (W_Q[i], W_K[i], W_V[i]):
                nn.init.xavier_uniform_(weight)
        self.W_Q = nn.ParameterList(W_Q)
        self.W_K = nn.ParameterList(W_K)
        self.W_V = nn.ParameterList(W_V)
        self.attention_layer = ScaledDotProductAttention(temperature = 1/math.sqrt(self.d_k))

    def forward(self, input, input_mask):
        Q, K, V = [], [], []
        heads = []
        for i in range(self.num_heads):
            Q.append(torch.matmul(input, self.W_Q[i]))
            K.append(torch.matmul(input, self.W_K[i]))
            V.append(torch.matmul(input, self.W_V[i]))
        for i in range(self.num_heads):
            heads.append(self.attention_layer(Q[i], K[i], V[i], input_mask)[0])
        concatenated = torch.cat(heads, dim=2)
        output = torch.matmul(concatenated, self.W_O)
        return output


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


# class DCN (nn.Module):
#     def __init__(self, hidden_size, drop_prob=0.1):
#         super(BiDAFAttention, self).__init__()
#         self.drop_prob = drop_prob
#         self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
#         self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
#         self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
#         for weight in (self.c_weight, self.q_weight, self.cq_weight):
#             nn.init.xavier_uniform_(weight)
#         self.bias = nn.Parameter(torch.zeros(1))
#
#
#     """
#     Here we will attempt to impliment a DCN model. Amanda and I are still a
#     little confused about what that means, but hopefully it works out.
#     """
#     def __init__(self, hidden_dim, maxout_pool_size, emb_matrix, max_dec_steps, dropout_ratio=0.1):
#         super(CoattentionModel, self).__init__()
#         self.hidden_dim = hidden_dim
#
#         self.encoder = RNNEncoder(2*hidden_dim, hidden_dim, 1, dropout_ratio)
#
#         self.q_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.fusion_bilstm = FusionBiLSTM(hidden_dim, dropout_ratio)
#         self.decoder = DynamicDecoder(hidden_dim, maxout_pool_size, max_dec_steps, dropout_ratio)
#         self.dropout = nn.Dropout(p=dropout_ratio)
#
#     def forward(self, q_seq, q_mask, d_seq, d_mask, span=None):
#         Q = self.encoder(q_seq, q_mask) # b x n + 1 x l
#         D = self.encoder(d_seq, d_mask)  # B x m + 1 x l
#
#         #project q
#         Q = F.tanh(self.q_proj(Q.view(-1, self.hidden_dim))).view(Q.size()) #B x n + 1 x l
#
#         #co attention
#         D_t = torch.transpose(D, 1, 2) #B x l x m + 1
#         L = torch.bmm(Q, D_t) # L = B x n + 1 x m + 1
#
#         A_Q_ = F.softmax(L, dim=1) # B x n + 1 x m + 1
#         A_Q = torch.transpose(A_Q_, 1, 2) # B x m + 1 x n + 1
#         C_Q = torch.bmm(D_t, A_Q) # (B x l x m + 1) x (B x m x n + 1) => B x l x n + 1
#
#         Q_t = torch.transpose(Q, 1, 2)  # B x l x n + 1
#         A_D = F.softmax(L, dim=2)  # B x n + 1 x m + 1
#         C_D = torch.bmm(torch.cat((Q_t, C_Q), 1), A_D) # (B x l x n+1 ; B x l x n+1) x (B x n +1x m+1) => B x 2l x m + 1
#
#         C_D_t = torch.transpose(C_D, 1, 2)  # B x m + 1 x 2l
#
#         #fusion BiLSTM
#         bilstm_in = torch.cat((C_D_t, D), 2) # B x m + 1 x 3l
#         bilstm_in = self.dropout(bilstm_in)
#         #?? should it be d_lens + 1 and get U[:-1]
#         U = self.fusion_bilstm(bilstm_in, d_mask) #B x m x 2l
#
#         loss, idx_s, idx_e = self.decoder(U, d_mask, span)
#         if span is not None:
#             return loss, idx_s, idx_e
#         else:
#             return idx_s, idx_e
#
# class FusionBiLSTM(nn.Module):
#     def __init__(self, hidden_dim, dropout_ratio):
#         super(FusionBiLSTM, self).__init__()
#         self.fusion_bilstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True,
#                                      bidirectional=True, dropout=dropout_ratio)
#         init_lstm_forget_bias(self.fusion_bilstm)
#         self.dropout = nn.Dropout(p=dropout_ratio)
#
#     def forward(self, seq, mask):
#         lens = torch.sum(mask, 1)
#         lens_sorted, lens_argsort = torch.sort(lens, 0, True)
#         _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
#         seq_ = torch.index_select(seq, 0, lens_argsort)
#         packed = pack_padded_sequence(seq_, lens_sorted, batch_first=True)
#         output, _ = self.fusion_bilstm(packed)
#         e, _ = pad_packed_sequence(output, batch_first=True)
#         e = e.contiguous()
#         e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
#         e = self.dropout(e)
#         return e

class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        # print('similarity')
        # print(c.shape)
        # print(q.shape)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)
        # print(c.shape)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class ModelEncoder(nn.Module):

    def __init__(self, kernel_size=7, d_model=96, num_layers=2, num_blocks=7, drop_prob=0.2, device=None):
        super(ModelEncoder, self).__init__()
        self.block = num_blocks * [EmbeddingEncoder(d_model=d_model, num_layers=num_layers, drop_prob=drop_prob)]

    def forward(self, input, mask):
        result = input
        for i in range(len(self.block)):
            result = self.block[i](result, mask)
        M1 = result
        result = M1
        for i in range(len(self.block)):
            result = self.block[i](result, mask)
        M2 = result
        result = M2
        for i in range(len(self.block)):
            result = self.block[i](result, mask)
        M3 = result
        return M1, M2, M3

class QANet(nn.Module):

    def __init__(self, hidden_size=96):
        super(QANet, self).__init__()
        self.layer_one = nn.Linear(hidden_size, 1)
        self.layer_two = nn.Linear(hidden_size, 1)

    def forward(self, M1, M2, M3, mask):
        first = torch.cat((M1, M2), dim=1) # (64, c_len+q_len, 96)
        second = torch.cat((M1, M3), dim=1)# (64, c_len+q_len, 96)

        # print('in hereeeeeee')
        # print(first.shape)
        # print(second.shape)

        logits_1 = self.layer_one(first)
        logits_2 = self.layer_two(second)

        # print(logits_1.shape)
        # print(logits_2.shape)
        # print(mask.shape)
        mask = torch.cat((mask, mask), dim=1)
        # print(mask.shape)

        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2



class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
