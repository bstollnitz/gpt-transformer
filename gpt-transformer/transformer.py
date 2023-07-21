# MIT License
# Copyright (c) 2018 Alexander Rush
# Original source:
# http://nlp.seas.harvard.edu/annotated-transformer/
# https://github.com/harvardnlp/annotated-transformer/

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math, copy

# batch_size     = batch size (1 in our simple inference scenario)
# input_seq_len  = length of the input sequence of tokens (4 in our scenario)
# seq_len        = length of the input sequence of tokens sent to the model, which is no larger than block_size (same as input_seq_len in our scenario)
# vocab_size     = number of tokens in the vocabulary (50257 in our scenario)
# d_model        = embedding length (we'll keep the default of 512)
# h              = number of heads (we'll keep the default of 8)
# d_k            = head size (512 / 8 = 64 in our scenario)
# N              = number of decoder copies (we'll keep the default of 6)
# d_ff           = dimension of inner layer of feed forward network (we'll keep the default of 2048)
# block_size     = length of the block to be decoded (1024 in our scenario)


# Used to convert the tokens into vectors of dimension d_model.
class Embeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    # input x: (batch_size, seq_len)
    # output: (batch_size, seq_len, d_model)
    def forward(self, x):
        out = self.lut(x) * math.sqrt(self.d_model)
        return out


# The goal of positional encoding is to add information about the relative or
# absolute position of the tokens in the sequence.
# The vectors produced by this layer are added to the embedding vectors.
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        # The sin and cos functions have the following frequency:
        # Frequency = 10000^(2i/d_model)
        # log(Frequency) = 2i * (-log(10000) / d_model)
        # Frequency = e^log(Frequency) = e^(2i * (-log(10000) / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model))  # (d_model/2)
        # Even columns of pe contain sine waves, and odd columns contain cosine
        # waves.
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    # input x: (batch_size, seq_len, d_model)
    # output: (batch_size, seq_len, d_model)
    def forward(self, x):
        # Truncate pe to match the dimensions of x before adding them.
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# Dimensions of query, key, and value: (batch_size, h, seq_len, d_k)
# Dimensions of mask: (1, 1, seq_len, seq_len)
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """

    d_k = query.size(-1)
    # (batch_size, h, seq_len, d_k) x (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
    # This does a dot product between all the queries and all the keys.
    # We divide by sqrt(d_k) to keep the variance of the scores at around 1.
    # Keeping the variance of scores low prevents the softmax operation
    # later on from returning one-hot vectors.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # (batch_size, h, seq_len, seq_len)
    # After the softmax, "p_attn" contains probabilities of how related each
    # token is to all other other tokens in the sequence, or the affinities
    # between tokens.
    if dropout is not None:
        p_attn = dropout(p_attn)
    # The multiplication of "p_attn" with "value" results in a weighted sum of
    # the the rows of "value", where the weights are the probabilities in
    # "p_attn".
    # (batch_size, h, seq_len, seq_len) x (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # (1, 1, seq_len, seq_len)

        nbatches = query.size(0)  # batch_size

        # 1) Do all the linear projections in batch from
        # (batch_size, seq_len, d_model) => (batch_size, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # This is equivalent to:
        # query = self.linears[0](query)
        # key = self.linears[1](key)
        # value = self.linears[2](value)
        # query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # key = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # value = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        # (batch_size, h, seq_len, d_k)
        x, self.attn = attention(query,
                                 key,
                                 value,
                                 mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # (batch_size, h, seq_len, d_k) => (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    # Matrix containing 1s in its upper triangle, excluding the diagonal. The
    # rest of the matrix contains 0s.
    subsequent_mask = torch.triu(torch.ones(attn_shape),
                                 diagonal=1).type(torch.uint8)
    # Matrix with True in its diagonal and lower triangle, and False in
    # its upper triangle.
    return subsequent_mask == 0


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class DecoderLayer(nn.Module):
    """
    Decoder is made of a self-attention layer and a feed forward layer.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class DecoderModel(nn.Module):
    """
    A standard Decoder Transformer architecture. Base for this and many 
    other models.
    """

    def __init__(self, decoder, embed, generator):
        super(DecoderModel, self).__init__()
        self.embed = embed
        self.decoder = decoder
        self.generator = generator

    def forward(self, x, mask):
        return self.decode(x, mask)

    def decode(self, x, mask):
        return self.decoder(self.embed(x), mask)


def make_model(vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = DecoderModel(
        Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
