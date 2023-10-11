import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
import numpy as np;

np.random.seed(0)


class Convdown(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convd = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))

        self.attn = ESSAttn(dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))  # + x_embed
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convd(x)
        x = x + shortcut
        return x


class Convup(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convu = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))
        self.drop = nn.Dropout2d(0.2)
        self.norm = nn.LayerNorm(dim)
        self.attn = ESSAttn(dim)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convu(x)
        x = x + shortcut
        return x


class blockup(nn.Module):
    def __init__(self, dim, upscale):
        super(blockup, self).__init__()
        self.convup = Convup(dim)
        self.convdown = Convdown(dim)
        self.convupsample = Upsample(scale=upscale, num_feat=dim)
        self.convdownsample = Downsample(scale=upscale, num_feat=dim)

    def forward(self, x):
        xup = self.convupsample(x)
        x1 = self.convup(xup)
        xdown = self.convdownsample(x1) + x
        x2 = self.convdown(xdown)
        xup = self.convupsample(x2) + x1
        x3 = self.convup(xup)
        xdown = self.convdownsample(x3) + x2
        x4 = self.convdown(xdown)
        xup = self.convupsample(x4) + x3
        x5 = self.convup(xup)
        return x5


class PatchEmbed(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class ESSAttn(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)
        # t2 = self.norm1(t2)*0.3
        # print(torch.mean(t1),torch.std(t1))
        # print(torch.mean(t2), torch.std(t2))
        # t2 = self.norm1(t2)*0.1
        # t2 = ((q2 / (q2s+1e-7)) @ t2)

        # q3 = torch.pow(q,4)
        # q3s = torch.pow(q2s,2)
        # k3 = torch.pow(k, 4)
        # k3s = torch.sum(k2,dim=2).unsqueeze(2).repeat(1, 1, C)
        # t3 = ((k3 / k3s)*16).transpose(-2, -1) @ v
        # t3 = ((q3 / q3s)*16) @ t3
        # print(torch.max(t1))
        # print(torch.max(t2))
        # t3 = (((torch.pow(q,4))/24) @ (((torch.pow(k,4).transpose(-2,-1))/24)@v)*16/math.sqrt(N))
        attn = t1 + t2
        attn = self.ln(attn)
        return attn

    def is_same_matrix(self, m1, m2):
        rows, cols = len(m1), len(m1[0])
        for i in range(rows):
            for j in range(cols):
                if m1[i][j] != m2[i][j]:
                    return False
        return True


class Downsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, num_feat // 9, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class ESSA(nn.Module):
    def __init__(self, inch, dim, upscale, **kwargs):
        super(ESSA, self).__init__()
        self.conv_first = nn.Conv2d(inch, dim, 3, 1, 1)
        self.blockup = blockup(dim=dim, upscale=upscale)
        self.conv_last = nn.Conv2d(dim, inch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.blockup(x)
        x = self.conv_last(x)
        return x


if __name__ == '__main__':
    upscale = 4
    height = 128
    width = 128
    model = ESSA(inch=128, dim=256, upscale=4)
    print(model)
    x = torch.randn((1, 128, height, width))
    x = model(x)
    print(x.shape)
