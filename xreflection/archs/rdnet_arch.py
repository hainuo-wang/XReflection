# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from xreflection.utils.registry import ARCH_REGISTRY
from xreflection.archs.rdnet.classifier import PretrainedConvNextV2
from xreflection.archs.rdnet.focalnet import build_focalnet
from xreflection.archs.rdnet.modules_sig import ConvNextBlock, Decoder, LayerNorm, NAFBlock, SimDecoder, UpSampleConvnext
from xreflection.archs.rdnet.revcol_function import ReverseFunction
from xreflection.models.model_zoo import prepare_model_path

class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
        ) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args
        channels_dowm = c_down.size(1)
        # print(channels_dowm)
        # c_down_split= torch.split(c_down, int(channels_dowm/2), -3)
        # c_down_clean=c_down_split[0]
        # c_down_refl=c_down_split[1]
        if self.first_col:
            x_clean = self.down(c_down)
            return x_clean  # ch.cat([x_clean, x_refl], dim=-3)
        if c_up is not None:
            channels_up = c_up.size(1)

        # c_up_clean, c_up_refl = torch.split(c_up, 2, -3)

        if self.level == 3:
            x_clean = self.down(c_down)
        else:
            x_clean = self.up(c_up) + self.down(c_down)

        return x_clean  # orch.cat([x_clean, x_refl], dim=-3)


class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, block_type=ConvNextBlock) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col)
        modules = [block_type(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
                              layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer + i]) for i in
                   range(layers[level])]
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x = self.fusion(*args)
        # x_clean, x_refl = torch.split(x, x.shape[-3] // 2, dim=-3
        x_clean = self.blocks(x)
        return x_clean


class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory,
                 block_type=ConvNextBlock) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        c0 = self.alpha0 * c0 + self.level0(x, c1)
        c1 = self.alpha1 * c1 + self.level1(c0, c2)
        c2 = self.alpha2 * c2 + self.level2(c1, c3)
        c3 = self.alpha3 * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):
        x, c0, c1, c2, c3 = args
        # [print(it.shape) if type(it) is not int else None for it in args]
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

##-------------------------------------- Tiny -----------------------------------------
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=True):
        super().__init__()
        # self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


@ARCH_REGISTRY.register()
class RDNet(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], layers=[2, 3, 6, 3], num_subnet=5, loss_col=4, kernel_size=3,
                 num_classes=1000,
                 drop_path=0.0, save_memory=True, 
                 inter_supv=True, head_init_scale=None, 
                 pretrained_cols=16,
                 pretrained_models=None,
                 ) -> None:
        super().__init__()

        self.num_subnet = num_subnet
        self.Loss_col = (loss_col + 1)

        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]
        
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                channels, layers, kernel_size, first_col,
                dp_rates=dp_rate, save_memory=save_memory,
                block_type=NAFBlock))

        channels.reverse()
        self.decoder_blocks = Decoder(depth=[1, 1, 1, 1], dim=channels, block_type=NAFBlock, kernel_size=3)

        self.prompt = nn.Sequential(nn.Conv2d(in_channels=768,
                                              out_channels=512,
                                              kernel_size=1),
                                    StarReLU(),
                                    nn.Conv2d(512, 64, kernel_size=1)
                                    )
        pretrained_cls_path = pretrained_models.pop('cls_model', None)
        pretrained_base_network_path = pretrained_models.pop('base_network', None)
        
        self.baseball_adapter = nn.ModuleList()
        self.baseball_adapter.append(nn.Conv2d(192, 64, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192, 64, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 2, 64 * 2, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 4, 64 * 4, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 8, 64 * 8, kernel_size=1))


        self.baseball = build_focalnet('focalnet_L_384_22k_fl4')
        self.classifier = PretrainedConvNextV2()
        
        for param in self.classifier.parameters():
            param.requires_grad = False

            
        if pretrained_cls_path is not None:
            pretrained_cls_path = prepare_model_path(pretrained_cls_path)
            self.classifier.load_state_dict(torch.load(pretrained_cls_path, map_location='cpu')['icnn'], strict=False)

        if pretrained_base_network_path is not None:
            pretrained_base_network_path = prepare_model_path(pretrained_base_network_path)
            self.baseball.load_state_dict(torch.load(pretrained_base_network_path, map_location='cpu'))

    def get_optimizer_params(self):
        # Setup different parameter groups with their learning rates
        train_opt = self.opt['train']
        params_lr = [
            {'params': self.net_g.get_baseball_params(), 'lr': train_opt['optim_g']['baseball_lr']},
            {'params': self.net_g.get_other_params(), 'lr': train_opt['optim_g']['other_lr']},
        ]

        # Get optimizer configuration without modifying original config
        optim_type = train_opt['optim_g']['type']
        optim_config = {k: v for k, v in train_opt['optim_g'].items()
                        if k not in ['type', 'baseball_lr', 'other_lr']}

        return {
            'params': params_lr,
            'type': optim_type,
            **optim_config
        }

    def forward(self, x_in, prompt=True):

        x_cls_out = []
        x_img_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        interval = self.num_subnet // 4

        x_base, x_stem = self.baseball(x_in)
        c0, c1, c2, c3 = x_base
        x_stem = self.baseball_adapter[0](x_stem)
        c0, c1, c2, c3 = self.baseball_adapter[1](c0), \
            self.baseball_adapter[2](c1), \
            self.baseball_adapter[3](c2), \
            self.baseball_adapter[4](c3)
        # x_in = x
        with torch.no_grad():
            self.classifier.eval()
            prompt = self.classifier(x_in)
        prompt = self.prompt(prompt)
        x = prompt * x_stem
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
            if i > (self.num_subnet - self.Loss_col):
                x_img_out.append(torch.cat([x_in, x_in], dim=-3) - self.decoder_blocks(c3, c2, c1, c0))
        return x_cls_out, x_img_out

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)

    def get_baseball_params(self):
        return self.baseball.parameters()

    def get_other_params(self):
        # get all params except for the baseball
        return list(self.prompt.parameters()) + list(self.decoder_blocks.parameters()) + \
            list(self.subnet0.parameters()) + list(self.subnet1.parameters()) + list(self.subnet2.parameters()) + list(
                self.subnet3.parameters()) + \
            list(self.baseball_adapter.parameters())


if __name__ == '__main__':
    import time

    device = torch.device('cuda:1')
    inp = torch.rand((1, 3, 1280, 1280), device=device)
    channels = [128, 256, 512, 1024]
    layers = [1, 2, 6, 2]
    num_subnet = 8
    model = FullNet(channels, layers, num_subnet, num_classes=1000, drop_path=0.4, save_memory=True, inter_supv=True,
                    head_init_scale=None, kernel_size=7).to(device).eval()
    torch.cuda.synchronize(device)
    out = model(inp)
    print(out[1][0].shape)

