import torch
import torch.nn as nn
import torch.nn.functional as F
from xreflection.utils.registry import LOSS_REGISTRY
from .vgg import Vgg19


###############################################################################
# Functions
###############################################################################
def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


@LOSS_REGISTRY.register()
class DSITExclusionLoss(nn.Module):
    """Exclusion loss for reflection separation.
    
    This loss encourages the transmission and reflection layers to be independent
    by minimizing their correlation.
    """

    def __init__(self, level=3, eps=1e-6):
        super(DSITExclusionLoss, self).__init__()
        self.level = level
        self.eps = eps

    def forward(self, pred_t, pred_r):
        """Forward function.
        
        Args:
            pred_t (Tensor): Predicted transmission layer.
            pred_r (Tensor): Predicted reflection layer.
            
        Returns:
            Tensor: Exclusion loss value.
        """
        grad_x_loss = []
        grad_y_loss = []

        for l in range(self.level):
            grad_x_T, grad_y_T = compute_gradient(pred_t)
            grad_x_R, grad_y_R = compute_gradient(pred_r)

            alphax = (2.0 * torch.mean(torch.abs(grad_x_T))) / (torch.mean(torch.abs(grad_x_R)) + self.eps)
            alphay = (2.0 * torch.mean(torch.abs(grad_y_T))) / (torch.mean(torch.abs(grad_y_R)) + self.eps)

            gradx1_s = (torch.sigmoid(grad_x_T) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
            grady1_s = (torch.sigmoid(grad_y_T) * 2) - 1
            gradx2_s = (torch.sigmoid(grad_x_R * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grad_y_R * alphay) * 2) - 1

            grad_x_loss.append(((torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2)))) + self.eps) ** 0.25)
            grad_y_loss.append(((torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2)))) + self.eps) ** 0.25)

            pred_t = F.interpolate(pred_t, scale_factor=0.5, mode='bilinear')
            pred_r = F.interpolate(pred_r, scale_factor=0.5, mode='bilinear')
        loss_gradxy = torch.sum(sum(grad_x_loss) / 3) + torch.sum(sum(grad_y_loss) / 3)

        return loss_gradxy / 2


@LOSS_REGISTRY.register()
class DSITReconsLoss(nn.Module):
    """Reconstruction loss for DSIT model.
    
    This loss ensures that the sum of transmission and reflection layers
    reconstructs the input image.
    """

    def __init__(self, ):
        super(DSITReconsLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred_t, pred_r, recons, input_img):
        """Forward function.
        
        Args:
            pred_t (Tensor): Predicted transmission layer.
            pred_r (Tensor): Predicted reflection layer.
            recons (Tensor): Reconstructed image.
            input_img (Tensor): Input image.
            
        Returns:
            Tensor: Reconstruction loss value.
        """
        content_diff = self.criterion(pred_t + pred_r + recons, input_img)
        return content_diff


@LOSS_REGISTRY.register()
class DSITPerceptualLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(DSITPerceptualLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss
