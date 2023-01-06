import math
import torch
import torch.nn as nn

MAX_TEXT_LEN = 144


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SinePositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(SinePositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].requires_grad_(False)
