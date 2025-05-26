import torch.nn as nn
import timm
import torch
import torch.nn.functional as F



def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    out_dict = {}
    if 'visual.trunk.stem.0.weight' in state_dict:
        out_dict = {k.replace('visual.trunk.', ''): v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        if 'visual.head.proj.weight' in state_dict:
            out_dict['head.fc.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
        elif 'visual.head.mlp.fc1.weight' in state_dict:
            out_dict['head.pre_logits.fc.weight'] = state_dict['visual.head.mlp.fc1.weight']
            out_dict['head.pre_logits.fc.bias'] = state_dict['visual.head.mlp.fc1.bias']
            out_dict['head.fc.weight'] = state_dict['visual.head.mlp.fc2.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.mlp.fc2.weight'].shape[0])
        return out_dict

    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        if 'grn' in k:
            k = k.replace('grn.beta', 'mlp.grn.bias')
            k = k.replace('grn.gamma', 'mlp.grn.weight')
            v = v.reshape(v.shape[-1])
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict

class PretrainedConvNext(nn.Module):
    def __init__(self, model_name='convnext_base', pretrained=True):
        super(PretrainedConvNext, self).__init__()
        # Load the pretrained ConvNext model from timm
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.head = nn.Linear(768, 6)
    def forward(self, x):
        with torch.no_grad():
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        # Forward pass through the ConvNext model
        out = self.model(cls_input)
        out = self.head(out)
        # alpha, beta = out[..., :3].unsqueeze(-1).unsqueeze(-1),\
        #               out[..., 3:].unsqueeze(-1).unsqueeze(-1)

        #out = alpha * x + beta
        # print(out.shape)
        return out#alpha,beta#out #out[..., :3], out[..., 3:]
class PretrainedConvNext_e2e(nn.Module):
    def __init__(self, model_name='convnext_base', pretrained=True):
        super(PretrainedConvNext_e2e, self).__init__()
        # Load the pretrained ConvNext model from timm
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        ret = self.model.load_state_dict(checkpoint_filter_fn(torch.load('./convnext_small_22k_224.pth'), self.model), strict=False)
        print(ret)
        self.head = nn.Linear(768, 6)
    def forward(self, x):
        with torch.no_grad():
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        # Forward pass through the ConvNext model
        out = self.model(cls_input)
        out = self.head(out)
        alpha, beta = out[..., :3].unsqueeze(-1).unsqueeze(-1),\
                      out[..., 3:].unsqueeze(-1).unsqueeze(-1)

        out = alpha * x + beta
        #print(out.shape)
        return out#alpha,beta#out #out[..., :3], out[..., 3:]


class PretrainedConvNextV2(nn.Module):
    def __init__(self, model_name='convnext_small', pretrained=True):
        super(PretrainedConvNextV2, self).__init__()
        # Load the pretrained ConvNext model from timm
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        # self.head = nn.Linear(768, 6)
    def forward(self, x):
        with torch.no_grad():
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        # Forward pass through the ConvNext model
        out = self.model.forward_features(cls_input)

        return F.interpolate(out, size=(x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=True)

if __name__ == "__main__":
    model = PretrainedConvNext('convnext_small_in22k')
    print("Testing PretrainedConvNext model...")
    # Assuming a dummy input tensor of size (1, 3, 224, 224) similar to an image in the ImageNet dataset
    dummy_input = torch.randn(20, 3, 224, 224)
    output_x, output_y = model(dummy_input)
    print("Output shape:", output_x.shape)
    print("Test completed successfully.")
