import numpy as np
from typing import Literal, Union

import torch
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef
from torch import autograd, nn, Tensor

def reduce(
    loss: Union[torch.Tensor, np.ndarray], 
    mask: Union[torch.Tensor, np.ndarray] = None, 
    reduction: Literal['mean', 'mean_in_mask', 'sum', 'max', 'min', 'none']='mean'):

    if mask is not None:
        if mask.dim() == loss.dim() - 1:
            mask = mask.view(*loss.shape[:-1], 1).expand_as(loss)
        assert loss.dim() == mask.dim(), f"Expects loss.dim={loss.dim()} to be equal to mask.dim()={mask.dim()}"
    
    if reduction == 'mean':
        return loss.mean() if mask is None else (loss * mask).mean()
    elif reduction == 'mean_in_mask':
        return loss.mean() if mask is None else (loss * mask).sum() / mask.sum().clip(1e-5)
    elif reduction == 'sum':
        return loss.sum() if mask is None else (loss * mask).sum()
    elif reduction == 'max':
        return loss.max() if mask is None else loss[mask].max()
    elif reduction == 'min':
        return loss.min() if mask is None else loss[mask].min()
    elif reduction == 'none':
        return loss if mask is None else loss * mask
    else:
        raise RuntimeError(f"Invalid reduction={reduction}")

class SafeBCE(autograd.Function):
    """ Perform clipped BCE without disgarding gradients (preserve clipped gradients)
        This function is equivalent to torch.clip(x, limit), 1-limit) before BCE, 
        BUT with grad existing on those clipped values.
        
    NOTE: pytorch original BCELoss implementation is equivalent to limit = np.exp(-100) here.
        see doc https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """
    @staticmethod
    def forward(ctx, x, y, limit):
        assert (torch.where(y!=1, y+1, y)==1).all(), u'target must all be {0,1}'
        ln_limit = ctx.ln_limit = np.log(limit)
        # ctx.clip_grad = clip_grad
        
        # NOTE: for example, torch.log(1-torch.tensor([1.000001])) = nan
        x = torch.clip(x, 0, 1)
        y = torch.clip(y, 0, 1)
        ctx.save_for_backward(x, y)
        return -torch.where(y==0, torch.log(1-x).clamp_min_(ln_limit), torch.log(x).clamp_min_(ln_limit))
        # return -(y * torch.log(x).clamp_min_(ln_limit) + (1-y)*torch.log(1-x).clamp_min_(ln_limit))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        ln_limit = ctx.ln_limit
        
        # NOTE: for y==0, do not clip small x; for y==1, do not clip small (1-x)
        limit = np.exp(ln_limit)
        # x = torch.clip(x, eclip, 1-eclip)
        x = torch.where(y==0, torch.clip(x, 0, 1-limit), torch.clip(x, limit, 1))
        
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            # ttt = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # with open('grad.txt', 'a') as fp:
            #     fp.write(f"{ttt.min().item():.05f}, {ttt.max().item():.05f}\n")
            # NOTE: " * (~(x==y))" so that those already match will not generate gradients.
            grad_x = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # grad_x = ( (1-y)/(1-x) - y/x ) * grad_output
        if ctx.needs_input_grad[1]:
            grad_y = (torch.log(1-x) - torch.log(x)) * grad_output * (~(x==y))
        #---- x, y, limit
        return grad_x, grad_y, None

def safe_binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, limit: float = 0.1, reduction="mean") -> torch.Tensor:
    loss = SafeBCE.apply(input, target, limit)
    return reduce(loss, None, reduction=reduction)

def binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, reduction="mean") -> torch.Tensor:
    loss = F.binary_cross_entropy(input, target, reduction="none")
    return reduce(loss, None, reduction=reduction)

def normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 0.0, 1.0)

def safe_normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 1e-06, 1.0)

class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_normal: Tensor,
        gt_normal: Tensor,
        mask: Tensor = None,
    ):
        # masked L1 loss
        loss = F.l1_loss(pred_normal, gt_normal, reduction="none")
        loss = reduce(loss, mask, reduction="mean_in_mask")
        return loss

class AlbedoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_albedo: Tensor,
        gt_albedo: Tensor,
        mask: Tensor = None,
    ):
        # NOTE: GT albedo is in sRGB space. We've already convert rendered albedo to sRGB space before loss computation.
        # masked L1 loss
        loss = F.l1_loss(pred_albedo, gt_albedo, reduction="none")
        loss = reduce(loss, mask, reduction="mean_in_mask")
        return loss

class AlbedoTVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_albedo: Tensor,
        mask: Tensor = None,
    ):
        # masked TV loss
        # (H, W, 3) -> (3, H, W)
        pred_albedo = pred_albedo.permute(2, 0, 1)
        
        # (H, W) / (H, W, 1) -> (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.permute(2, 0, 1)
        else:
            raise RuntimeError(f"Invalid mask dim: {mask.dim()}")
        
        tv_h = torch.pow(pred_albedo[:, 1:, :] - pred_albedo[:, :-1, :], 2)  # [3, H-1, W]
        tv_w = torch.pow(pred_albedo[:, :, 1:] - pred_albedo[:, :, :-1], 2)  # [3, H, W-1]
        mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
        mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]
        
        loss = (tv_h * mask_h).mean() + (tv_w * mask_w).mean()
        
        return loss

class MetallicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_metallic: Tensor,
        gt_metallic: Tensor,
        mask: Tensor = None,
    ):
        # masked L1 loss
        loss = F.l1_loss(pred_metallic, gt_metallic, reduction="none")
        loss = reduce(loss, mask, reduction="mean_in_mask")
        return loss

class MetallicTVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_metallic: Tensor,
        mask: Tensor = None,
    ):
        # masked TV loss
        # (H, W, 1) -> (1, H, W)
        pred_metallic = pred_metallic.permute(2, 0, 1)
        
        # (H, W) / (H, W, 1) -> (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.permute(2, 0, 1)
        else:
            raise RuntimeError(f"Invalid mask dim: {mask.dim()}")
        
        tv_h = torch.pow(pred_metallic[:, 1:, :] - pred_metallic[:, :-1, :], 2)  # [1, H-1, W]
        tv_w = torch.pow(pred_metallic[:, :, 1:] - pred_metallic[:, :, :-1], 2)  # [1, H, W-1]
        mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
        mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]
        
        loss = (tv_h * mask_h).mean() + (tv_w * mask_w).mean()
        
        return loss

class RoughnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_roughness: Tensor,
        gt_roughness: Tensor,
        mask: Tensor = None,
    ):
        # masked L1 loss
        loss = F.l1_loss(pred_roughness, gt_roughness, reduction="none")
        loss = reduce(loss, mask, reduction="mean_in_mask")
        return loss

class RoughnessTVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_roughness: Tensor,
        mask: Tensor = None,
    ):
        # masked TV loss
        # (H, W, 1) -> (1, H, W)
        pred_roughness = pred_roughness.permute(2, 0, 1)
        
        # (H, W) / (H, W, 1) -> (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.permute(2, 0, 1)
        else:
            raise RuntimeError(f"Invalid mask dim: {mask.dim()}")
        
        tv_h = torch.pow(pred_roughness[:, 1:, :] - pred_roughness[:, :-1, :], 2)  # [1, H-1, W]
        tv_w = torch.pow(pred_roughness[:, :, 1:] - pred_roughness[:, :, :-1], 2)  # [1, H, W-1]
        mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
        mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]
        
        loss = (tv_h * mask_h).mean() + (tv_w * mask_w).mean()
        
        return loss

class NormalCosLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_normal: Tensor,
        gt_normal: Tensor,
        mask: Tensor = None,
    ):
        # masked cosine similarity loss
        pred_normal = F.normalize(pred_normal, p=2, dim=2, eps=1e-6)
        gt_normal = F.normalize(gt_normal, p=2, dim=2, eps=1e-6)
        loss = 1 - (pred_normal * gt_normal).sum(dim=2).clamp(-1, 1)
        loss = reduce(loss, mask, reduction="mean_in_mask")
        return loss

# Adapted from GS-IR: https://github.com/lzhnb/GS-IR
class NormalTVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_normal: Tensor,
        gt_rgb: Tensor,
        mask: Tensor = None,
    ):
        # masked TV loss
        # (H, W, 3) -> (3, H, W)
        gt_rgb = gt_rgb.permute(2, 0, 1)
        pred_normal = pred_normal.permute(2, 0, 1)
        
        # (H, W) / (H, W, 1) -> (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.permute(2, 0, 1)
        else:
            raise RuntimeError(f"Invalid mask dim: {mask.dim()}")
        
        gt_grad_h = torch.exp(
            -(gt_rgb[:, 1:, :] - gt_rgb[:, :-1, :]).abs().mean(dim=0, keepdim=True)
        )  # [1, H-1, W]
        gt_grad_w = torch.exp(
            -(gt_rgb[:, :, 1:] - gt_rgb[:, :, :-1]).abs().mean(dim=0, keepdim=True)
        )
        tv_h = torch.pow(pred_normal[:, 1:, :] - pred_normal[:, :-1, :], 2)  # [3, H-1, W]
        tv_w = torch.pow(pred_normal[:, :, 1:] - pred_normal[:, :, :-1], 2)  # [3, H, W-1]
        mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
        mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]
        
        loss = (tv_h * gt_grad_h * mask_h).mean() + (tv_w * gt_grad_w * mask_w).mean()
        
        return loss

class DepthTVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_depth: Tensor,
        gt_rgb: Tensor,
        mask: Tensor = None,
    ):
        # masked TV loss with edge-aware weighting
        # (H, W, 1) -> (1, H, W)
        pred_depth = pred_depth.permute(2, 0, 1)
        
        # (H, W, 3) -> (3, H, W)
        gt_rgb = gt_rgb.permute(2, 0, 1)
        
        # (H, W) / (H, W, 1) -> (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.permute(2, 0, 1)
        else:
            raise RuntimeError(f"Invalid mask dim: {mask.dim()}")
        
        gt_grad_h = torch.exp(
            -(gt_rgb[:, 1:, :] - gt_rgb[:, :-1, :]).abs().mean(dim=0, keepdim=True)
        )  # [1, H-1, W]
        gt_grad_w = torch.exp(
            -(gt_rgb[:, :, 1:] - gt_rgb[:, :, :-1]).abs().mean(dim=0, keepdim=True)
        )
        tv_h = torch.pow(pred_depth[:, 1:, :] - pred_depth[:, :-1, :], 2)  # [1, H-1, W]
        tv_w = torch.pow(pred_depth[:, :, 1:] - pred_depth[:, :, :-1], 2)  # [1, H, W-1]
        mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
        mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]
        
        loss = (tv_h * gt_grad_h * mask_h).mean() + (tv_w * gt_grad_w * mask_w).mean()
        
        return loss

class RoadNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_normal: Tensor,
        road_mask: Tensor,
    ):
        # compute the mean normal vector, and minimize the 1 - cosine similarity of all normal vectors to the mean normal vector
        masked_normals = pred_normal[road_mask.bool()]  # [N, 3]
        # assume that all normal vectors are normalized
        mean_normal = F.normalize(masked_normals.mean(dim=0, keepdim=True), p=2, dim=1, eps=1e-6)  # [1, 3]
        cos_sim = (masked_normals * mean_normal).sum(dim=1).clamp(-1, 1)  # [N]
        loss = (1 - cos_sim).mean()
        return loss
        

def compute_scale_and_shift(prediction, target):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(prediction * prediction)
    a_01 = torch.sum(prediction)
    ones = torch.ones_like(prediction)
    a_11 = torch.sum(ones)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(prediction * target)
    b_1 = torch.sum(target)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    # x_0 = torch.zeros_like(b_0)
    # x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        x_0 = torch.FloatTensor(0).cuda()
        x_1 = torch.FloatTensor(0).cuda()

    return x_0, x_1

def compute_shift(prediction, target, weight=None):
    """
    Compute the best shift to align prediction to target.
    Args:
        prediction (torch.Tensor): predicted values.
        target (torch.Tensor): target values.
        weight (torch.Tensor, optional): weight for each element. Should be the same shape as prediction and target.
    Returns:
        shift (torch.Tensor): computed shift value.
    """
    # Use mean difference between target and prediction for least-squares shift
    diff = target - prediction
    if weight is not None:
        weighted_diff = diff * weight
        shift = weighted_diff.sum() / weight.sum().clip(1e-5)
    else:
        shift = diff.mean()
    return shift

class MonoDepth2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor, 
        lidar_depth: Tensor,
        max_depth: float = 80, 
    ):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze() # monocular depth
        lidar_depth  = lidar_depth.squeeze()
        
        lidar_hit_mask = (lidar_depth > 0.0)

        # negative depth (conf < 0.1 in pi3) will be filtered out here
        valid_mask = (gt_depth > 0.01) & (gt_depth < max_depth) & (pred_depth > 0.0001)

        valid_hit_mask = valid_mask & lidar_hit_mask

        # align monocular depth (gt depth) with lidar depth by scale and shift

        lidar_gt_depth = gt_depth[valid_hit_mask]
        lidar_filtered_depth = lidar_depth[valid_hit_mask]
        scale, shift = compute_scale_and_shift(lidar_gt_depth, lidar_filtered_depth)
        
        aligned_gt_depth = gt_depth * scale + shift

        # L1
        loss = torch.abs(pred_depth[valid_mask] - aligned_gt_depth[valid_mask])
        dist_decay = torch.exp(-pred_depth.detach()[valid_mask])
        loss = (loss * dist_decay).mean()
        
        # # scale and median
        # if sky_mask is not None:
        #     pred_depth = pred_depth[~(sky_mask.bool())]
        #     gt_depth = gt_depth[~(sky_mask.bool())]
        # # use disparity
        # pred_depth = 1./pred_depth
        # gt_depth = 1./gt_depth
        # t_d = torch.median(pred_depth)
        # s_d = torch.mean(torch.abs(pred_depth - t_d))
        # pred_depth = (pred_depth - t_d) / s_d
        # t_gt = torch.median(gt_depth)
        # s_gt = torch.mean(torch.abs(gt_depth - t_gt))
        # gt_depth = (gt_depth - t_gt) / s_gt
        # loss = (pred_depth - gt_depth) ** 2
        # loss = loss.mean()
        
        # # pearson correlation coefficient
        # if sky_mask is not None:
        #     pred_depth = pred_depth[~(sky_mask.bool())]
        #     gt_depth = gt_depth[~(sky_mask.bool())]
        # pred_depth = pred_depth.reshape(-1, 1)
        # gt_depth = gt_depth.reshape(-1, 1)
        # loss = min(
        #     1 - pearson_corrcoef(gt_depth, pred_depth),
        #     1 - pearson_corrcoef(1 / (gt_depth + 200), pred_depth)
        # )
        # loss = loss.mean()
        
        return loss

class MonoDepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor, 
        sky_mask: Tensor = None,
        extra_loss_mask: Tensor = None, # only compute mono depth loss in this region
        max_depth: float = 80, 
    ):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()
        
        # negative depth (conf < 0.1 in pi3) will be filtered out here
        valid_mask = (gt_depth > 0.01) & (gt_depth < max_depth) & (pred_depth > 0.0001)

        pred_depth = pred_depth[valid_mask]
        gt_depth = gt_depth[valid_mask]
        if sky_mask is not None:
            sky_mask = sky_mask[valid_mask]
        if extra_loss_mask is not None:
            extra_loss_mask = extra_loss_mask[valid_mask].float()
        else:
            extra_loss_mask = torch.ones_like(pred_depth).float()
        
        # scale-shift invariant
        if sky_mask is not None:
            scale, shift = compute_scale_and_shift(pred_depth.detach()[~(sky_mask.bool())], gt_depth[~(sky_mask.bool())])
        else:
            scale,  shift = compute_scale_and_shift(pred_depth.detach(), gt_depth)
        pred_depth = pred_depth * scale + shift
        loss = (pred_depth - gt_depth) ** 2
        dist_decay = torch.exp(-pred_depth.detach())
        loss = (loss * dist_decay * extra_loss_mask).mean()
        
        # # scale and median
        # if sky_mask is not None:
        #     pred_depth = pred_depth[~(sky_mask.bool())]
        #     gt_depth = gt_depth[~(sky_mask.bool())]
        # # use disparity
        # pred_depth = 1./pred_depth
        # gt_depth = 1./gt_depth
        # t_d = torch.median(pred_depth)
        # s_d = torch.mean(torch.abs(pred_depth - t_d))
        # pred_depth = (pred_depth - t_d) / s_d
        # t_gt = torch.median(gt_depth)
        # s_gt = torch.mean(torch.abs(gt_depth - t_gt))
        # gt_depth = (gt_depth - t_gt) / s_gt
        # loss = (pred_depth - gt_depth) ** 2
        # loss = loss.mean()
        
        # # pearson correlation coefficient
        # if sky_mask is not None:
        #     pred_depth = pred_depth[~(sky_mask.bool())]
        #     gt_depth = gt_depth[~(sky_mask.bool())]
        # pred_depth = pred_depth.reshape(-1, 1)
        # gt_depth = gt_depth.reshape(-1, 1)
        # loss = min(
        #     1 - pearson_corrcoef(gt_depth, pred_depth),
        #     1 - pearson_corrcoef(1 / (gt_depth + 200), pred_depth)
        # )
        # loss = loss.mean()
        
        return loss
        
        
class DepthLoss(nn.Module):
    def __init__(
        self,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        normalize: bool = True,
        use_inverse_depth: bool = False,
        depth_error_percentile: float = None,
        upper_bound: float = 80,
        reduction: Literal["mean_on_hit", "mean_on_hw", "sum", "none"] = "mean_on_hit",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.use_inverse_depth = use_inverse_depth
        self.upper_bound = upper_bound
        self.depth_error_percentile = depth_error_percentile
        self.reduction = reduction

    def _compute_depth_loss(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        max_depth: float = 80,
        hit_mask: Tensor = None,
    ):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()
        if hit_mask is not None:
            pred_depth = pred_depth * hit_mask
            gt_depth = gt_depth * hit_mask
        
        # cal valid mask to make sure gt_depth is valid
        valid_mask = (gt_depth > 0.01) & (gt_depth < max_depth) & (pred_depth > 0.0001)
        
        # normalize depth to (0, 1)
        if self.normalize:
            pred_depth = safe_normalize_depth(pred_depth[valid_mask], max_depth=max_depth)
            gt_depth = safe_normalize_depth(gt_depth[valid_mask], max_depth=max_depth)
        else:
            pred_depth = pred_depth[valid_mask]
            gt_depth = gt_depth[valid_mask]
        
        # inverse the depth map (0, 1) -> (1, +inf)
        if self.use_inverse_depth:
            pred_depth = 1./pred_depth
            gt_depth = 1./gt_depth
            
        # cal loss
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l2":
            return F.mse_loss(pred_depth, gt_depth, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        hit_mask: Tensor = None,
    ):
        depth_error = self._compute_depth_loss(pred_depth, gt_depth, self.upper_bound, hit_mask)
        if self.depth_error_percentile is not None:
            # to avoid outliers. not used for now
            depth_error = depth_error.flatten()
            depth_error = depth_error[
                depth_error.argsort()[
                    : int(len(depth_error) * self.depth_error_percentile)
                ]
            ]
        
        if self.reduction == "sum":
            depth_error = depth_error.sum()
        elif self.reduction == "none":
            depth_error = depth_error
        elif self.reduction == "mean_on_hit":
            # depth_error = depth_error.mean()
            # safe mean
            depth_error = depth_error.sum() / max(depth_error.numel(), 1)
        elif self.reduction == "mean_on_hw":
            n = gt_depth.shape[0]*gt_depth.shape[1]
            depth_error = depth_error.sum() / n
        else:
            raise NotImplementedError(f"Unknown reduction method: {self.reduction}")

        return depth_error