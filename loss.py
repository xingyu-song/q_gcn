import torch
import torch.nn.functional as F

def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Mean Per-Joint Position Error
    pred, gt: (B, T, N, 3)
    """
    return (pred - gt).norm(dim=-1).mean()

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def quat_geodesic_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """
    Geodesic-ish quaternion loss using |dot|:
    q_pred, q_gt: (B, T, Bn, 4)
    """
    q_pred = quat_normalize(q_pred)
    q_gt = quat_normalize(q_gt)
    # q and -q represent the same rotation, so use abs(dot)
    dot = (q_pred * q_gt).sum(dim=-1).abs().clamp(0.0, 1.0)
    # angle = 2*acos(dot); minimize angle
    return (2.0 * torch.acos(dot)).mean()

def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Simple temporal smoothness (first difference).
    x: (B, T, ..., D)
    """
    return (x[:, 1:] - x[:, :-1]).abs().mean()