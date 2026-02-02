import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from model.py import mpjpe, quat_geodesic_loss, smoothness_loss

@dataclass
class TrainConfig:
    device: str = "cuda"
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    grad_clip: float = 1.0
    use_amp: bool = True

    # loss weights (tune these)
    w_pose: float = 1.0
    w_quat: float = 0.2
    w_smooth_pose: float = 0.0
    w_smooth_quat: float = 0.0

    ckpt_dir: str = "./checkpoints"
    ckpt_name: str = "qgcn.pt"


def save_checkpoint(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer,
                    epoch: int, best_val: float, extra: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module, optim: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return ckpt


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_pose = 0.0
    total_quat = 0.0
    n = 0

    for batch in loader:
        P2d = batch["P2d"].to(device)         # (B,T,N,2)
        R2d = batch["R2d"].to(device)         # (B,T,Bn,2)
        P3d_gt = batch["P3d_gt"].to(device)   # (B,T',N,3)
        Q_gt = batch.get("Q_gt", None)
        if Q_gt is not None:
            Q_gt = Q_gt.to(device)            # (B,T',Bn,4)

        P3d_pred, Q_pred = model(P2d, R2d)

        # Make sure GT is aligned to pred
        # If your dataset already matches T', this should just work.
        if P3d_gt.shape[1] != P3d_pred.shape[1]:
            raise ValueError(f"GT T'={P3d_gt.shape[1]} != pred T'={P3d_pred.shape[1]}")

        pose_l = mpjpe(P3d_pred, P3d_gt)

        quat_l = torch.tensor(0.0, device=device)
        if Q_gt is not None:
            if Q_gt.shape[1] != Q_pred.shape[1]:
                raise ValueError(f"Quat GT T'={Q_gt.shape[1]} != pred T'={Q_pred.shape[1]}")
            quat_l = quat_geodesic_loss(Q_pred, Q_gt)

        bsz = P2d.shape[0]
        total_pose += pose_l.item() * bsz
        total_quat += quat_l.item() * bsz
        n += bsz

    return {
        "pose_mpjpe": total_pose / max(n, 1),
        "quat_loss": total_quat / max(n, 1),
    }


def train(model, train_loader: DataLoader, val_loader: Optional[DataLoader], cfg: TrainConfig):
    device = cfg.device
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.startswith("cuda")))

    best_val = float("inf")
    start_epoch = 0

    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, model, optim)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {ckpt_path} @ epoch={start_epoch}, best_val={best_val:.6f}")

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        running = 0.0
        num = 0

        for batch in train_loader:
            P2d = batch["P2d"].to(device)
            R2d = batch["R2d"].to(device)
            P3d_gt = batch["P3d_gt"].to(device)
            Q_gt = batch.get("Q_gt", None)
            if Q_gt is not None:
                Q_gt = Q_gt.to(device)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                P3d_pred, Q_pred = model(P2d, R2d)

                if P3d_gt.shape[1] != P3d_pred.shape[1]:
                    raise ValueError(f"GT T'={P3d_gt.shape[1]} != pred T'={P3d_pred.shape[1]}")

                pose_l = mpjpe(P3d_pred, P3d_gt)

                quat_l = torch.tensor(0.0, device=device)
                if Q_gt is not None:
                    if Q_gt.shape[1] != Q_pred.shape[1]:
                        raise ValueError(f"Quat GT T'={Q_gt.shape[1]} != pred T'={Q_pred.shape[1]}")
                    quat_l = quat_geodesic_loss(Q_pred, Q_gt)

                smooth_pose = smoothness_loss(P3d_pred) if cfg.w_smooth_pose > 0 else torch.tensor(0.0, device=device)
                smooth_quat = smoothness_loss(Q_pred) if cfg.w_smooth_quat > 0 else torch.tensor(0.0, device=device)

                loss = (
                    cfg.w_pose * pose_l
                    + cfg.w_quat * quat_l
                    + cfg.w_smooth_pose * smooth_pose
                    + cfg.w_smooth_quat * smooth_quat
                )

            scaler.scale(loss).backward()

            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optim)
            scaler.update()

            bsz = P2d.shape[0]
            running += loss.item() * bsz
            num += bsz

        train_loss = running / max(num, 1)

        log = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device=device)
            val_key = metrics["pose_mpjpe"]  # pick your "best" criterion
            log.update(metrics)

            is_best = val_key < best_val
            if is_best:
                best_val = val_key

            save_checkpoint(ckpt_path, model, optim, epoch=epoch, best_val=best_val)
        else:
            save_checkpoint(ckpt_path, model, optim, epoch=epoch, best_val=best_val)

        print(log)

    return model