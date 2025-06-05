import torch
import torch.nn as nn
import torch.nn.functional as F
from xreflection.utils.registry import ARCH_REGISTRY
from collections import OrderedDict
from timm.models.layers import to_2tuple, trunc_normal_

from xreflection.archs.swin_det import swin_large_384_det
from xreflection.models.model_zoo import prepare_model_path


def window_partition(x, window_size):
    if not isinstance(window_size, tuple):
        window_size = (window_size, window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    if not isinstance(window_size, tuple):
        window_size = (window_size, window_size)

    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SinBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2),
            SimpleGate(),
            CABlock(c),
            nn.Conv2d(c, c, 1)
        )

        self.block2 = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            SimpleGate(),
            nn.Conv2d(c, c, 1)
        )

        self.a = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.block1(inp)
        x_skip = inp + x * self.a
        x = self.block2(x_skip)
        out = x_skip + x * self.b
        return out


class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)


class MuGIBlock(nn.Module):
    def __init__(self, c, shared_b=True):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class LayeredWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, num_layers=2, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * 3, num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_l = torch.arange(num_layers)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_l, coords_h, coords_w]))  # 3, Wl, Wh, Ww,
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*2

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wl, Wh*Ww*Wl
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wl, Wh*Ww*Wl, 3
        relative_coords[:, :, 0] += 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[0] - 1
        relative_coords[:, :, 2] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wl, Wh*Ww*Wl
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * 2, self.window_size[0] * self.window_size[1] * 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DualAttentionInteractiveBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=12, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)

        self.norm1 = DualStreamBlock(norm_layer(dim))
        self.dsia_sa = WindowAttention(dim, to_2tuple(self.window_size), num_heads=num_heads)
        self.dsia_ca = LayeredWindowAttention(dim, to_2tuple(self.window_size), num_heads=num_heads)

        self.feedforward = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(dim),
                nn.Conv2d(dim, dim * 2, 1),
                nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(dim)),
            DualStreamBlock(nn.Conv2d(dim, dim, 1))
        )

        self.a = nn.Parameter(torch.zeros((1, 1, dim)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x, y):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        y = y.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        x_skip, y_skip = x, y
        x, y = self.norm1(x, y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
        if not self.training:
            # pad feature maps to multiples of window size
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape
        else:
            Hp, Wp = H, W
            pad_r, pad_b = 0, 0

        x_windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        y_windows = window_partition(y, self.window_size).view(-1, self.window_size * self.window_size, C)

        xx_windows, yy_windows = self.dsia_sa(torch.cat([x_windows, y_windows], dim=0)).chunk(2, dim=0)
        xy_windows, yx_windows = self.dsia_ca(torch.cat([x_windows, y_windows], dim=-2)).chunk(2, dim=-2)

        x_windows = (xx_windows + xy_windows).view(-1, self.window_size, self.window_size, C)
        y_windows = (yy_windows + yx_windows).view(-1, self.window_size, self.window_size, C)

        x = window_reverse(x_windows, self.window_size, Hp, Wp)  # B H' W' C
        y = window_reverse(y_windows, self.window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            y = y[:, :H, :W, :].contiguous()

        x = x_skip + x.view(B, H * W, C) * self.a
        y = y_skip + y.view(B, H * W, C) * self.a

        x_skip = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        y_skip = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x, y = self.feedforward(x_skip, y_skip)
        x, y = x_skip + x * self.b, y_skip + y * self.b
        return x, y


class LocalFeatureExtractor(nn.Module):
    def __init__(self, dims, enc_blk_nums=[]):
        super(LocalFeatureExtractor, self).__init__()
        self.dims = dims

        c = dims
        self.stem = DualStreamBlock(nn.Conv2d(3, c, 3, padding=1))
        self.block1 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[0])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2
        self.block2 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[1])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2
        self.block3 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[2])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2
        self.block4 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[3])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2
        self.block5 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[4])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )

    def forward(self, x):
        # (1, 48, 384, 384) (1, 96, 192, 192) (1, 192, 96, 96) (1, 384, 48, 48) (1, 768, 24, 24) (1, 1536, 12, 12)
        x0, y0 = self.stem(x, x)
        x1, y1 = self.block1(x0, y0)
        x2, y2 = self.block2(x1, y1)
        x3, y3 = self.block3(x2, y2)
        x4, y4 = self.block4(x3, y3)
        x5, y5 = self.block5(x4, y4)
        return (x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)


@ARCH_REGISTRY.register()
class DSIT(nn.Module):
    def __init__(self, pretrained_models=None, input_resolution=(384, 384), window_size=12, enc_blk_nums=[],
                 dec_blk_nums=[]):
        super().__init__()
        pretrained_swin_prior_path = pretrained_models.pop('swin_prior', None)
        if pretrained_swin_prior_path is not None:
            pretrained_swin_prior_path = prepare_model_path(pretrained_swin_prior_path)
        self.swin_prior = swin_large_384_det(pretrained_swin_prior_path)
        self.swin_prior.eval()
        self.swin_prior.requires_grad_(False)
        self.conv_prior = LocalFeatureExtractor(48, [2, 2, 2, 2, 2])

        self.input_resolution = input_resolution
        self.device = 'cuda'
        self.window_size = window_size
        H, W = input_resolution
        self.aib5 = DualStreamSeq(
            DualStreamBlock(nn.PixelShuffle(2)),
            DualAttentionInteractiveBlock(384, (H // 16, W // 16), 8, window_size),
            DualStreamBlock(nn.Conv2d(in_channels=384, out_channels=768, kernel_size=1)),

        )

        self.aib4 = DualStreamSeq(
            DualAttentionInteractiveBlock(768, (H // 16, W // 16), 8, window_size)
        )

        self.lib4 = DualStreamSeq(
            *[DualAttentionInteractiveBlock(768, (H // 16, W // 16), 8, window_size)
              for _ in range(enc_blk_nums[0])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(192) for _ in range(dec_blk_nums[0])],
            DualStreamBlock(nn.Conv2d(in_channels=192, out_channels=384, kernel_size=1))
        )

        self.aib3 = DualStreamSeq(
            DualAttentionInteractiveBlock(384, (H // 8, W // 8), 8, window_size),
        )

        self.lib3 = DualStreamSeq(
            *[DualAttentionInteractiveBlock(384, (H // 8, W // 8), 8, window_size)
              for _ in range(enc_blk_nums[1])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(96) for _ in range(dec_blk_nums[1])],
            DualStreamBlock(nn.Conv2d(in_channels=96, out_channels=192, kernel_size=1))
        )

        self.aib2 = DualStreamSeq(
            DualAttentionInteractiveBlock(192, (H // 4, W // 4), 4, window_size),
        )

        self.lib2 = DualStreamSeq(
            *[DualAttentionInteractiveBlock(192, (H // 4, W // 4), 4, window_size)
              for _ in range(enc_blk_nums[2])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(48) for _ in range(dec_blk_nums[2])],
            DualStreamBlock(nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1))
        )

        self.lib1 = DualStreamSeq(
            *[DualAttentionInteractiveBlock(96, (H // 2, W // 2), 2, window_size)
              for _ in range(enc_blk_nums[3])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(24) for _ in range(dec_blk_nums[3])],
            DualStreamBlock(nn.Conv2d(in_channels=24, out_channels=48, kernel_size=1))
        )

        self.lib0 = DualStreamSeq(
            *[DualAttentionInteractiveBlock(48, (H, W), 1, window_size)
              for _ in range(enc_blk_nums[4])],
            *[MuGIBlock(48) for _ in range(dec_blk_nums[4])],
        )

        self.out = DualStreamBlock(nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, padding=1))
        self.lrm = nn.Sequential(
            SinBlock(48),
            nn.Conv2d(48, 3, 3, padding=1),
            nn.Tanh()
        )

    def train(self, mode=True):
        super().train(mode)
        self.swin_prior.eval()

    def forward(self, inp, fn=None):
        # (1, 192, 96, 96), (1, 384, 48, 48), (1, 768, 24, 24), (1, 768, 12, 12)
        sp2, sp3, sp4, sp5 = self.swin_prior(inp)
        # (1, 48, 384, 384), (1, 96, 192, 192), (1, 192, 96, 96), (1, 384, 48, 48), (1, 768, 24, 24), (1, 1536, 12, 12)
        (cp0_l, cp0_r), (cp1_l, cp1_r), (cp2_l, cp2_r), (cp3_l, cp3_r), (cp4_l, cp4_r), (
            cp5_l, cp5_r) = self.conv_prior(inp)

        scp5_l, csp5_l = self.aib5(sp5, cp5_l)
        scp5_r, csp5_r = self.aib5(sp5, cp5_r)

        scp4_l, csp4_l = self.aib4(sp4, cp4_l)
        scp4_r, csp4_r = self.aib4(sp4, cp4_r)

        f4_l, f4_r = self.lib4(scp5_l + csp5_l + scp4_l + csp4_l, scp5_r + csp5_r + scp4_r + csp4_r)

        scp3_l, csp3_l = self.aib3(sp3, cp3_l)
        scp3_r, csp3_r = self.aib3(sp3, cp3_r)

        f3_l, f3_r = self.lib3(f4_l + scp3_l + csp3_l, f4_r + scp3_r + csp3_r)

        scp2_l, csp2_l = self.aib2(sp2, cp2_l)
        scp2_r, csp2_r = self.aib2(sp2, cp2_r)

        f2_l, f2_r = self.lib2(f3_l + scp2_l + csp2_l, f3_r + scp2_r + csp2_r)
        f1_l, f1_r = self.lib1(f2_l + cp1_l, f2_r + cp1_r)
        f0_l, f0_r = self.lib0(f1_l + cp0_l, f1_r + cp0_r)
        out_l, out_r = self.out(f0_l, f0_r)
        out_rr = self.lrm(f0_l + f0_r)

        return out_l, out_r, out_rr


if __name__ == '__main__':
    inp = torch.randn(1, 3, 384, 384).cuda()
    model = DSIT(pretrained_models={'swin_prior': 'swin_large_384_det.pth'},
                 input_resolution=(384, 384),
                 window_size=12,
                 enc_blk_nums=[2, 2, 2, 2, 2],
                 dec_blk_nums=[2, 2, 2, 2, 2]).cuda()
    print(model(inp)[0].shape)
