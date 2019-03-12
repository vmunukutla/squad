"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from cnn import CNN


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
        input_shape = y.shape
        input_reshaped = torch.reshape(y, (input_shape[0] * input_shape[1], input_shape[2]))
        x_padded = self.char_embed(input_reshaped)
        x_reshaped = x_padded.permute(0, 2, 1)
        x_conv_out = self.cnn.forward(x_reshaped)

        x_highway = self.highway(x_conv_out)
        x_highway_reshaped = torch.reshape(x_highway, (input_shape[0], input_shape[1], x_highway.shape[1]))
        x_word_emb = F.dropout(x_highway_reshaped, self.drop_prob, self.training)

        word_emb = self.word_embed(x)   # (batch_size, seq_len, embed_size)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        word_emb = torch.cat((word_emb, x_word_emb), dim=2)
        word_emb = self.proj(word_emb)  # (batch_size, seq_len, hidden_size)
        word_emb = self.hwy(word_emb)   # (batch_size, seq_len, hidden_size)

        return word_emb

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


class EmbeddingEncoder(nn.Module):

    def __init__(self, embed_size, kernel_size, num_filters, num_layers):
        self.conv_layers = [nn.Conv1d(embed_size, num_filters, kernel_size) for i in range(num_layers))]
        self.layer_norm = [nn.LayerNorm(normalized_shape) for i in range(num_layers)+2]
        self.attention =


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


class DCN (nn.Module):
    """
    Here we will attempt to impliment a DCN model. Amanda and I are still a
    little confused about what that means, but hopefully it works out.
    """
    def __init__(self, hidden_dim, maxout_pool_size, emb_matrix, max_dec_steps, dropout_ratio):
        super(CoattentionModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = RNNEncoder(2*hidden_dim, hidden_dim, 1, dropout_ratio)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_bilstm = FusionBiLSTM(hidden_dim, dropout_ratio)
        self.decoder = DynamicDecoder(hidden_dim, maxout_pool_size, max_dec_steps, dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, q_seq, q_mask, d_seq, d_mask, span=None):
        Q = self.encoder(q_seq, q_mask) # b x n + 1 x l
        D = self.encoder(d_seq, d_mask)  # B x m + 1 x l

        #project q
        Q = F.tanh(self.q_proj(Q.view(-1, self.hidden_dim))).view(Q.size()) #B x n + 1 x l

        #co attention
        D_t = torch.transpose(D, 1, 2) #B x l x m + 1
        L = torch.bmm(Q, D_t) # L = B x n + 1 x m + 1

        A_Q_ = F.softmax(L, dim=1) # B x n + 1 x m + 1
        A_Q = torch.transpose(A_Q_, 1, 2) # B x m + 1 x n + 1
        C_Q = torch.bmm(D_t, A_Q) # (B x l x m + 1) x (B x m x n + 1) => B x l x n + 1

        Q_t = torch.transpose(Q, 1, 2)  # B x l x n + 1
        A_D = F.softmax(L, dim=2)  # B x n + 1 x m + 1
        C_D = torch.bmm(torch.cat((Q_t, C_Q), 1), A_D) # (B x l x n+1 ; B x l x n+1) x (B x n +1x m+1) => B x 2l x m + 1

        C_D_t = torch.transpose(C_D, 1, 2)  # B x m + 1 x 2l

        #fusion BiLSTM
        bilstm_in = torch.cat((C_D_t, D), 2) # B x m + 1 x 3l
        bilstm_in = self.dropout(bilstm_in)
        #?? should it be d_lens + 1 and get U[:-1]
        U = self.fusion_bilstm(bilstm_in, d_mask) #B x m x 2l

        loss, idx_s, idx_e = self.decoder(U, d_mask, span)
        if span is not None:
            return loss, idx_s, idx_e
        else:
            return idx_s, idx_e

class FusionBiLSTM(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio):
        super(FusionBiLSTM, self).__init__()
        self.fusion_bilstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True,
                                     bidirectional=True, dropout=dropout_ratio)
        init_lstm_forget_bias(self.fusion_bilstm)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, seq, mask):
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        packed = pack_padded_sequence(seq_, lens_sorted, batch_first=True)
        output, _ = self.fusion_bilstm(packed)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
        e = self.dropout(e)
        return e

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
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


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
