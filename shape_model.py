import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import copy
import itertools
import math


class Encoder(nn.Module):
    """
    Args:
        n_embed: dimension of representations
        n_hid: dimension of hidden layer
        n_head: number of head
        n_elayers: dimension of transformer.encoder
        n_mlayers: dimension of transformer.mattn
        n_dlayers: dimension of transformer.decoder
    """

    def __init__(self, n_point=3, n_embed=128, n_hid=512, n_head= 4, n_elayers = 6, dropout=0.1):
        super(Encoder, self).__init__()

        n_enco = 32

        self.encoder_pos = nn.Sequential(nn.Linear(n_point*2, n_enco, bias=False),  # pos = [208, 1, 6]
                                         nn.ReLU(),
                                         nn.LayerNorm(n_enco),
                                         nn.Linear(n_enco, n_embed, bias=False))      # e_pos = [208, bz, 128]

        self.encoder_c = nn.Sequential(nn.Linear(1, n_enco, bias=False),        # cap = [208， bz, 1]
                                       nn.ReLU(),
                                       nn.LayerNorm(n_enco),
                                       nn.Linear(n_enco, n_embed, bias=False))

        self.encoder_src = nn.Sequential(
            nn.LayerNorm(n_embed))

        encoder_layer = TransformerEncoderLayer(n_embed, n_head, n_hid, dropout)
        self.Transformer_encoder = TransformerEncoder(encoder_layer, n_elayers)

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sens_pos, vols, batch_size):
        """
        Args:
        src: the sequence to the encoder - cap
        tgt: the sequence to the decoder - points
        """
        embedding_pos = self.encoder_pos(sens_pos)  # [208, 1, 128]
        e_pos = embedding_pos.expand(embedding_pos.size(0), batch_size, embedding_pos.size(-1))
        src = self.encoder_src(self.encoder_c(vols) + e_pos)  # [208, 32, 128]
        mem = self.Transformer_encoder(src)           # [208, 32, 128]
        return mem


class Decoder(nn.Module):
    """
    Args:
        n_embed: dimension of representations
        n_hid: dimension of hidden layer
        n_head: number of head
        n_elayers: dimension of transformer.encoder
        n_mlayers: dimension of transformer.mattn
        n_dlayers: dimension of transformer.decoder
    """

    def __init__(self, n_point=3, n_embed=128, n_hid=128, n_head=4,
                 n_mlayers=1, n_dlayers= 5, dropout=0.1):
        super(Decoder, self).__init__()
        n_enco = 32
        self.encoder_p = nn.Sequential(nn.Linear(3, n_enco),  # [m, bz, 3]
                                       nn.ReLU(),
                                       nn.LayerNorm(n_enco),
                                       nn.Linear(n_enco, n_embed),
                                       nn.ReLU(),
                                       nn.LayerNorm(n_embed))

        self.decoder = nn.Sequential(nn.Linear(n_embed, n_point),
                                     nn.Tanh())

        decoder_layer = TransformerDecoderLayer(n_embed, n_head, n_hid, dropout)
        self.Transformer_decoder = TransformerDecoder(decoder_layer, n_dlayers)

        if n_mlayers is not None:
            mutual_attn_layer = MultualAttnLayer(n_embed, n_head, n_hid, dropout)
            self.Transformer_mattn = MutualAttn(mutual_attn_layer, n_mlayers)
            self.mattn = True

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def build_grid(self, batch_size, r):
        # ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in self.meshgrid])
        # ndim = len(self.meshgrid)
        # grid = np.zeros((np.prod([it[2] for it in self.meshgrid]), ndim), dtype=np.float32)  # MxD
        # for d in range(ndim):
        #     grid[:, d] = np.reshape(ret[d], -1)
        # theta = np.array([90 - 18 * i for i in range(1350)])
        # x = np.array([0 + r * math.cos(math.radians(theta[i])) for i in range(theta.shape[0])])
        # y = np.array([0 + r * math.sin(math.radians(theta[i])) for i in range(theta.shape[0])])

        # 1D grid point
        # x = np.linspace(-r, r, 206)
        # y = np.linspace(-r, r, 206)
        # grid = np.dstack([x, y])
        # grid = np.repeat(grid, repeats=batch_size, axis=0)
        # grid = torch.tensor(grid)

        # 2D grid points
        x = np.linspace(-r, r, 12)
        y = np.linspace(-r, r, 12)
        grid = np.array(list(itertools.product(x, y)))
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
        grid = torch.tensor(grid)

        return grid.float()

    def forward(self, mem, grid):
        """
        Args:
        src: the sequence to the encoder - cap
        tgt: the sequence to the decoder - points
        """
        # grid = self.build_grid(mem.shape[1], r=r).transpose(0, 1)  # [m, bz, 3]
        # if torch.cuda.is_available():
        #     grid = grid.cuda()

        tgt = self.encoder_p(grid)  # [m, bz, 128]
        if hasattr(self, "mattn"):
            tgt2 = self.Transformer_mattn(tgt, mem)

            output = 1.2*self.decoder(self.Transformer_decoder(tgt2, mem))
        else:
            output = 1.2*self.decoder(self.Transformer_decoder(tgt, mem))

        return output  # [m, bz, 3]


class Model(nn.Module):
    """
    Args:
        n_embed: dimension of representations
        n_hid: dimension of hidden layer
        n_head: number of head
        n_elayers: dimension of transformer.encoder
        n_mlayers: dimension of transformer.mattn
        n_dlayers: dimension of transformer.decoder
    """

    def __init__(self, n_point=3, n_embed=64, n_hid=128, n_head=4, n_elayers=1,
                 n_mlayers=1, n_dlayers=1, dropout=0.1):
        super(Model, self).__init__()

        self.encoder = Encoder(n_point, n_embed, n_hid, n_head, n_elayers, dropout)
        self.decoder1 = Decoder(n_point, n_embed, n_hid, n_head, n_mlayers, n_dlayers, dropout)
        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sensor_pos, vols, initial1, initial2, initial3, batch_size):
        """
        Args:
        src: the sequence to the encoder - cap
        tgt: the sequence to the decoder - points
        """
        mem = self.encoder(sensor_pos, vols, batch_size)
        output1 = self.decoder1(mem, initial1.permute(1, 0, 2)).permute(1, 0, 2)
        output2 = self.decoder1(mem, initial2.permute(1, 0, 2)).permute(1, 0, 2)
        output3 = self.decoder1(mem, initial3.permute(1, 0, 2)).permute(1, 0, 2)

        return output1, output2, output3        # [bz, m, 3]


class MutualAttn(nn.Module):
    """MutualAttn is a stack of N MutualAttn layers

    Args:
        ma_layer: an instance of MutualAttn layer
        n_mlayers: the number of layers

    """

    def __init__(self, ma_layer, n_mlayers):
        super(MutualAttn, self).__init__()
        self.layers = _get_clones(ma_layer, n_mlayers)
        self.num_layers = n_mlayers

    def forward(self, tgt, mem, mem_mask=None, mem_key_padding_mask=None):
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](tgt=output, mem=mem, mem_mask=mem_mask,
                                    mem_key_padding_mask=mem_key_padding_mask)

        return output


class MultualAttnLayer(nn.Module):
    def __init__(self, n_embed, n_head, n_hid, dropout=0.1, activation="relu"):
        super(MultualAttnLayer, self).__init__()
        self.mutual_attn = nn.MultiheadAttention(n_embed, n_head, dropout=dropout)
        # Implementation of feedforward model
        self.linear1 = nn.Linear(n_embed, n_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(n_hid, n_embed)

        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, mem, mem_mask=None, mem_key_padding_mask=None):
        tgt2 = self.mutual_attn(tgt, mem, mem, attn_mask=mem_mask,
                                key_padding_mask=mem_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    else:
        return F.gelu


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return max((1/1024)*loss_1, (1/1000)*loss_2)

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
