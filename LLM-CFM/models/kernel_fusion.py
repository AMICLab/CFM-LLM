import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiKernel(nn.Module):
    def __init__(self, dim, trainable=True):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.tensor(1.0 / dim), requires_grad=trainable)
        self.alpha     = nn.Parameter(torch.tensor(0.5), requires_grad=trainable)
        self.c         = nn.Parameter(torch.tensor(1.0), requires_grad=trainable)
        self.w         = nn.Parameter(torch.ones(3), requires_grad=trainable)

    def forward(self, x, y):
        gamma = torch.exp(self.log_gamma)
        dist  = torch.cdist(x, y, p=2) ** 2
        k_rbf = torch.exp(-gamma * dist)
        k_lin = torch.bmm(x, y.transpose(1, 2))
        k_poly = (self.alpha * k_lin + self.c).pow(2)
        k = self.w[0]*k_rbf + self.w[1]*k_lin + self.w[2]*k_poly
        return k / (self.w.sum() + 1e-8)

class KernelFusion(nn.Module):
    def __init__(self, z_ch=3, text_dim=512, hid=64):
        super().__init__()
        self.z_proj   = nn.Conv2d(z_ch, hid, 1)
        self.text_proj = nn.Linear(text_dim, hid)
        self.kernel   = MultiKernel(hid)
        self.out      = nn.Conv2d(hid, z_ch, 1)

    def forward(self, z, text_vec):
        B, C, H, W = z.shape
        z_map = self.z_proj(z)  # [B, hid, H, W]
        t_map = self.text_proj(text_vec)[:, :, None, None]  # [B, hid, 1, 1]

        # 把 z 拉成序列
        z_seq = z_map.flatten(2).transpose(1, 2)  # [B, H*W, hid]

        # 把 t 先压成 3 维，再展开，再转置
        t_map_3d = t_map.squeeze(-1)  # [B, hid, 1]
        t_seq = t_map_3d.expand(-1, -1, H * W).transpose(1, 2)  # [B, H*W, hid]

        # 多核融合
        K = self.kernel(z_seq, t_seq)  # [B, H*W, H*W]
        weight = K.diagonal(dim1=1, dim2=2).view(B, 1, H, W)  # [B, 1, H, W]

        z_fused = z_map * (1 + torch.sigmoid(weight))
        return self.out(z_fused)  # [B, z_ch, H, W]