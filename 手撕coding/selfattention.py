import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        """
        x: [B, T, C]
        attn_mask: [T, T] or [B, T, T]
            True 表示这个位置要被 mask 掉
        return:
            out: [B, T, C]
            attn: [B, T, T]
        """
        B, T, C = x.shape

        Q = self.q_proj(x)  # [B, T, C]
        K = self.k_proj(x)  # [B, T, C]
        V = self.v_proj(x)  # [B, T, C]

        # [B, T, C] @ [B, C, T] -> [B, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(C)

        if attn_mask is not None:
            # True 的地方不允许 attend
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B, T, T]

        out = torch.matmul(attn, V)       # [B, T, C]
        out = self.out_proj(out)          # [B, T, C]

        return out, attn



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [B, T, C]
        attn_mask:
            [T, T] or [B, T, T]
            True 表示 mask 掉
        return:
            out: [B, T, C]
            attn: [B, H, T, T]
        """
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        Q = self.q_proj(x)  # [B, T, C]
        K = self.k_proj(x)  # [B, T, C]
        V = self.v_proj(x)  # [B, T, C]

        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        Q = Q.view(B, T, H, D).transpose(1, 2)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)

        # [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

        if attn_mask is not None:
            # [T, T] -> [1, 1, T, T]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            # [B, T, T] -> [B, 1, T, T]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B, H, T, T]
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)       # [B, H, T, D]

        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(out)

        return out, attn
