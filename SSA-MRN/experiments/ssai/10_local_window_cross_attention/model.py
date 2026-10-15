"""SSAI 10: local-window cross attention in the SSA slot.

The original backbone, band fusion convolutions (conv48tnum*), PReLU and decoder
are unchanged. Only the per-band SSA computation is replaced. The original SSA
modules stay registered (unused) so the common initial state is byte-identical
to experiment 4 A0; new parameters live under ``ssai_local.*`` and are created
after the original network, so the original random initialisation is unchanged.

Arms
  L2  query = band input (current 4ch features + MS band), key = backbone 64ch
      features, value = PAN encoder; 7x7 window softmax (2 heads, width 16)
  L1  same as L2 but key = PAN encoder (original SSAI inputs only)
  C   parameter-matched convolution control: same inputs (band input, backbone
      key projection, PAN encoder) mixed by 1x1 -> depthwise 7x7 -> 1x1, no attention
Each band returns 4 channels so the original 16->4 fusion convolution is kept.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common.model import RestoredPansharpeningNet

STAGES = ('SSA_blocks', 'SSA_blocks1', 'SSA_blocks2')
WIDTH = 16
HEADS = 2
WINDOW = 7
BANDS = 4
CONTROL_WIDTH = 10


def half(x):
    # Same operator as the model's downsample100/downsample200.
    return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)


def _offsets(window):
    return [(dy, dx) for dy in range(window) for dx in range(window)]


def _valid(h, w, r, like):
    return F.pad(torch.ones(1, 1, h, w, device=like.device, dtype=like.dtype), (r, r, r, r))[0, 0]


class LocalWindowAttention(torch.autograd.Function):
    """Same mathematics as reference_local_window_attention with an explicit backward.

    Autograd through 49 shifted slices allocated a full zero tensor per slice in
    backward (slice/select backward), which dominated the step time. Here the
    gradients are accumulated in place into padded buffers. All reductions are
    plain sums, so the result stays deterministic.
    """

    @staticmethod
    def forward(ctx, q, k, v, bias, window):
        b, n, heads, d, h, w = q.shape
        r = window // 2
        kp = F.pad(k.reshape(b, heads * d, h, w), (r, r, r, r)).view(b, heads, d, h + 2 * r, w + 2 * r)
        vp = F.pad(v.reshape(b, heads * d, h, w), (r, r, r, r)).view(b, heads, d, h + 2 * r, w + 2 * r)
        valid = _valid(h, w, r, q)
        scale = d ** -0.5
        offsets = _offsets(window)
        logits = q.new_empty(b, n, heads, len(offsets), h, w)
        invalid = torch.stack([valid[dy:dy + h, dx:dx + w] for dy, dx in offsets], 0).view(1, 1, 1, len(offsets), h, w) == 0
        for o, (dy, dx) in enumerate(offsets):
            logits[:, :, :, o] = (q * kp[..., dy:dy + h, dx:dx + w].unsqueeze(1)).sum(3) * scale + bias[:, :, o].view(1, n, heads, 1, 1)
        attention = torch.softmax(logits.masked_fill_(invalid, float('-inf')), dim=3)
        del logits
        out = q.new_zeros(b, n, heads, d, h, w)
        for o, (dy, dx) in enumerate(offsets):
            out += attention[:, :, :, o].unsqueeze(3) * vp[..., dy:dy + h, dx:dx + w].unsqueeze(1)
        ctx.save_for_backward(q, kp, vp, attention)
        ctx.window, ctx.scale = window, scale
        return out

    @staticmethod
    def backward(ctx, grad):
        q, kp, vp, attention = ctx.saved_tensors
        window, scale = ctx.window, ctx.scale
        b, n, heads, d, h, w = q.shape
        r = window // 2
        offsets = _offsets(window)
        grad = grad.contiguous()
        grad_attention = torch.empty_like(attention)
        grad_vp = torch.zeros_like(vp)
        for o, (dy, dx) in enumerate(offsets):
            grad_attention[:, :, :, o] = (grad * vp[..., dy:dy + h, dx:dx + w].unsqueeze(1)).sum(3)
            grad_vp[..., dy:dy + h, dx:dx + w] += (attention[:, :, :, o].unsqueeze(3) * grad).sum(1)
        grad_logits = attention * (grad_attention - (grad_attention * attention).sum(3, keepdim=True))
        del grad_attention
        grad_bias = grad_logits.sum(dim=(0, 4, 5))
        grad_q = torch.zeros_like(q)
        grad_kp = torch.zeros_like(kp)
        for o, (dy, dx) in enumerate(offsets):
            scaled = grad_logits[:, :, :, o].unsqueeze(3) * scale
            grad_q += scaled * kp[..., dy:dy + h, dx:dx + w].unsqueeze(1)
            grad_kp[..., dy:dy + h, dx:dx + w] += (scaled * q).sum(1)
        grad_k = grad_kp[..., r:r + h, r:r + w].contiguous()
        grad_v = grad_vp[..., r:r + h, r:r + w].contiguous()
        return grad_q, grad_k, grad_v, grad_bias, None


def local_window_attention(q, k, v, bias, window=WINDOW):
    """q: (B,N,H,D,h,w); k,v: (B,H,D,h,w) shared by the N bands; bias: (N,H,window^2).

    Softmax runs over the valid neighbours of each pixel only (out-of-image
    positions masked), so interior outputs do not depend on the image size.
    Neighbours are read as shifted views; no 49x unfolded copy is materialised.
    """
    return LocalWindowAttention.apply(q, k, v, bias, window)


def reference_local_window_attention(q, k, v, bias, window=WINDOW):
    """Plain-autograd reference used by the preflight to check the explicit backward."""
    b, n, heads, d, h, w = q.shape
    r = window // 2
    kp = F.pad(k.reshape(b, heads * d, h, w), (r, r, r, r)).view(b, heads, d, h + 2 * r, w + 2 * r)
    vp = F.pad(v.reshape(b, heads * d, h, w), (r, r, r, r)).view(b, heads, d, h + 2 * r, w + 2 * r)
    valid = F.pad(torch.ones(1, 1, h, w, device=q.device, dtype=q.dtype), (r, r, r, r))
    scale = d ** -0.5
    logits, masks = [], []
    for dy in range(window):
        for dx in range(window):
            ks = kp[..., dy:dy + h, dx:dx + w].unsqueeze(1)
            logits.append((q * ks).sum(3) * scale)
            masks.append(valid[0, 0, dy:dy + h, dx:dx + w])
    logits = torch.stack(logits, 3) + bias.view(1, n, heads, window * window, 1, 1)
    invalid = torch.stack(masks, 0).view(1, 1, 1, window * window, h, w) == 0
    attention = torch.softmax(logits.masked_fill(invalid, float('-inf')), dim=3)
    out = None
    o = 0
    for dy in range(window):
        for dx in range(window):
            term = attention[:, :, :, o].unsqueeze(3) * vp[..., dy:dy + h, dx:dx + w].unsqueeze(1)
            out = term if out is None else out + term
            o += 1
    return out  # (B,N,H,D,h,w)


class PanEncoder(nn.Sequential):
    def __init__(self, width=WIDTH):
        super().__init__(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1))


class LocalAttentionStage(nn.Module):
    def __init__(self, key_source):
        super().__init__()
        if key_source not in ('backbone', 'pan'):
            raise ValueError(key_source)
        self.key_source = key_source
        self.pan_encoder = PanEncoder()
        self.key = nn.Conv2d(64 if key_source == 'backbone' else WIDTH, WIDTH, 1)
        self.query = nn.ModuleList(nn.Conv2d(BANDS + 1, WIDTH, 3, padding=1) for _ in range(BANDS))
        self.project = nn.ModuleList(nn.Conv2d(WIDTH, 4, 1) for _ in range(BANDS))
        self.position_bias = nn.Parameter(torch.zeros(BANDS, HEADS, WINDOW * WINDOW))

    def forward(self, pan, features, ms, backbone):
        value = self.pan_encoder(pan)
        key = self.key(backbone if self.key_source == 'backbone' else value)
        b, _, h, w = value.shape
        d = WIDTH // HEADS
        q = torch.stack([self.query[i](torch.cat([features, ms[:, i:i + 1]], 1)) for i in range(BANDS)], 1)
        attended = local_window_attention(q.view(b, BANDS, HEADS, d, h, w), key.view(b, HEADS, d, h, w),
                                          value.view(b, HEADS, d, h, w), self.position_bias)
        attended = attended.reshape(b, BANDS, WIDTH, h, w)
        return torch.cat([self.project[i](attended[:, i]) for i in range(BANDS)], 1)


class ConvolutionControlStage(nn.Module):
    """Same inputs and 7x7 extent as L2, convolution instead of attention."""

    def __init__(self, width=CONTROL_WIDTH):
        super().__init__()
        self.pan_encoder = PanEncoder()
        self.key = nn.Conv2d(64, WIDTH, 1)
        self.mix = nn.ModuleList(nn.Conv2d(BANDS + 1 + 2 * WIDTH, width, 1) for _ in range(BANDS))
        self.spatial = nn.ModuleList(nn.Conv2d(width, width, WINDOW, padding=WINDOW // 2, groups=width) for _ in range(BANDS))
        self.project = nn.ModuleList(nn.Conv2d(width, 4, 1) for _ in range(BANDS))

    def forward(self, pan, features, ms, backbone):
        value = self.pan_encoder(pan)
        key = self.key(backbone)
        outputs = []
        for i in range(BANDS):
            x = torch.relu(self.mix[i](torch.cat([features, ms[:, i:i + 1], key, value], 1)))
            outputs.append(self.project[i](self.spatial[i](x)))
        return torch.cat(outputs, 1)


class LocalWindowSSAINet(RestoredPansharpeningNet):
    def __init__(self, arm, channels=4, ssai_dimension=4):
        if channels != BANDS or ssai_dimension != 4:
            raise ValueError('QB C4 / original K4 fusion only')
        super().__init__(channels, ssai_dimension)
        self.arm = arm
        if arm == 'L2':
            stages = [LocalAttentionStage('backbone') for _ in STAGES]
        elif arm == 'L1':
            stages = [LocalAttentionStage('pan') for _ in STAGES]
        elif arm == 'C':
            stages = [ConvolutionControlStage() for _ in STAGES]
        else:
            raise ValueError(arm)
        self.ssai_local = nn.ModuleList(stages)
        self._stage_features = []
        self.backbone.register_forward_hook(self._keep_backbone)

    def _keep_backbone(self, module, inputs, output):
        self._stage_features = [output]

    def _attend(self, pan, features, ms, blocks, fusion, activation):
        index = next(i for i, name in enumerate(STAGES) if getattr(self, name) is blocks)
        while len(self._stage_features) <= index:
            self._stage_features.append(half(self._stage_features[-1]))
        backbone = self._stage_features[index]
        if backbone.shape[-2:] != pan.shape[-2:]:
            raise RuntimeError('Backbone feature size mismatch')
        out = self.ssai_local[index](pan, features, ms, backbone)
        if index == len(STAGES) - 1:
            self._stage_features = []
        return activation(fusion(out))


def common_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('ssai_local.')}


def new_parameter_count(model):
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith('ssai_local.'))
