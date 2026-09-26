import torch
import torch.nn as nn
import lpips


DEFAULT_RGB_INDEX = (3, 2, 1)
_LPIPS_CACHE = {}


def _normalize_device(device=None):
    if device is None:
        if torch.cuda.is_available():
            return torch.device(f'cuda:{torch.cuda.current_device()}')
        return torch.device('cpu')

    if not isinstance(device, torch.device):
        device = torch.device(device)

    if device.type == 'cuda' and device.index is None:
        return torch.device(f'cuda:{torch.cuda.current_device()}')
    return device


class LPIPSLoss(nn.Module):
    """
    感知损失模块，基于LPIPS实现
    用于计算多光谱图像中RGB通道的感知损失
    """
    
    def __init__(self, RGB_index=DEFAULT_RGB_INDEX, net='vgg'):
        """
        Args:
            RGB_index: 多光谱数据中RGB通道的索引，默认为[3, 2, 1]，对应Sentinel-2的B4, B3, B2波段
            net: LPIPS使用的特征提取网络，可选 'alex', 'vgg', 'squeeze'
        """
        super().__init__()
        self.RGB_index = tuple(RGB_index)
        self.lpips_model = lpips.LPIPS(net=net).eval()
        
        # 冻结lpips模型参数
        for param in self.lpips_model.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target, reduction='mean'):
        """
        计算感知损失
        Args:
            pred: 预测的多光谱图像 [B, C, H, W]
            target: 目标多光谱图像 [B, C, H, W]
        Returns:
            感知损失值
        """
        if pred.device != target.device:
            raise ValueError('pred and target must be on the same device for LPIPS computation')
        if pred.shape[1] <= max(self.RGB_index):
            raise ValueError(
                f'LPIPS RGB indices {self.RGB_index} exceed channel count {pred.shape[1]}'
            )
        
        # 提取RGB通道
        pred_rgb = pred[:, self.RGB_index, :, :]  # [B, 3, H, W]
        target_rgb = target[:, self.RGB_index, :, :]  # [B, 3, H, W]
        
        # 归一化到[-1, 1]范围（lpips要求的输入范围）
        pred_rgb = pred_rgb * 2.0 - 1.0
        target_rgb = target_rgb * 2.0 - 1.0
        
        # 计算lpips损失
        loss = self.lpips_model(pred_rgb, target_rgb).flatten(start_dim=1).mean(dim=1)

        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        raise ValueError(f'Unsupported reduction: {reduction}')


def get_lpips_loss(device=None, RGB_index=DEFAULT_RGB_INDEX, net='vgg'):
    device = _normalize_device(device)
    cache_key = (device.type, device.index if device.type == 'cuda' else -1,
                 tuple(RGB_index), net)

    if cache_key not in _LPIPS_CACHE:
        lpips_loss = LPIPSLoss(RGB_index=RGB_index, net=net).to(device)
        lpips_loss.eval()
        _LPIPS_CACHE[cache_key] = lpips_loss

    return _LPIPS_CACHE[cache_key]


@torch.no_grad()
def compute_lpips(pred, target, reduction='mean', RGB_index=DEFAULT_RGB_INDEX, net='vgg'):
    lpips_loss = get_lpips_loss(pred.device, RGB_index=RGB_index, net=net)
    return lpips_loss(pred, target, reduction=reduction)
