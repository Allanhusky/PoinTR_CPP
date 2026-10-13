"""Local spacing penalty with bounded-memory neighbor search."""
import math

import torch


class RepulsionLoss:
    def __init__(self, k=5, radius=.02, anchor_points=1024, chunk_size=256):
        for name, value, minimum in [('k', k, 1), ('anchor_points', anchor_points, 0), ('chunk_size', chunk_size, 1)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'repulsion.{name} must be an integer >= {minimum}')
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('repulsion.radius must be finite and positive')
        self.k, self.radius = k, radius
        self.anchor_points, self.chunk_size = anchor_points, chunk_size

    def __call__(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] <= self.k:
            raise ValueError('Repulsion expects [B,N,3] with N > k')
        batch, count, _ = points.shape
        # 锚点随机抽样，候选邻居包含所有 query 的生成点。
        anchors = (torch.randperm(count, device=points.device)[:self.anchor_points]
                   if 0 < self.anchor_points < count else torch.arange(count, device=points.device))
        total = points.new_zeros(())
        for chunk in anchors.split(self.chunk_size):
            # 不保存距离矩阵的梯度，只对选中邻居的距离求导。
            with torch.no_grad():
                distances = torch.cdist(points[:, chunk].detach().float(), points.detach().float())
                distances[:, torch.arange(len(chunk), device=points.device), chunk] = float('inf')
                neighbors = distances.topk(self.k, dim=-1, largest=False).indices
                del distances
            batch_indices = torch.arange(batch, device=points.device)[:, None, None]
            offsets = points[:, chunk, None, :] - points[batch_indices, neighbors]
            spacing = torch.linalg.vector_norm(offsets, dim=-1)
            total = total + torch.relu(1-spacing/self.radius).square().sum()
        return total/(batch*len(anchors)*self.k)


def repulsion_settings(config):
    """Old experiments remain unchanged when loss.repulsion is absent."""
    value = dict(config.get('loss', {}).get('repulsion', {}))
    allowed = {'enabled', 'k', 'radius', 'weight', 'warmup_epochs', 'anchor_points', 'chunk_size'}
    if value.keys()-allowed:
        raise ValueError(f'Unknown loss.repulsion parameters: {sorted(value.keys()-allowed)}')
    settings = dict(enabled=False, k=5, radius=.02, weight=.005,
                    warmup_epochs=20, anchor_points=1024, chunk_size=256)
    settings.update(value)
    if not isinstance(settings['enabled'], bool):
        raise ValueError('repulsion.enabled must be true or false')
    weight, warmup = settings['weight'], settings['warmup_epochs']
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('repulsion.weight must be finite and nonnegative')
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('repulsion.warmup_epochs must be an integer >= 0')
    loss = RepulsionLoss(**{key: settings[key] for key in ['k', 'radius', 'anchor_points', 'chunk_size']})
    return settings, loss


def repulsion_weight(settings, epoch):
    if not settings['enabled']:
        return 0.
    warmup = settings['warmup_epochs']
    return settings['weight']*(min((epoch+1)/warmup, 1.) if warmup else 1.)
