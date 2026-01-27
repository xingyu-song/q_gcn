import torch

@torch.no_grad()
def infer(model, P2d: torch.Tensor, R2d: torch.Tensor, device: str = "cuda"):
    """
    P2d: (B, T, N, 2)
    R2d: (B, T, Bn, 2)
    Returns:
      P3d: (B, T', N, 3)
      Q:   (B, T', Bn, 4)
    """
    model.eval()
    model = model.to(device)
    P2d = P2d.to(device)
    R2d = R2d.to(device)

    P3d, Q = model(P2d, R2d)

    # Optionally normalize quaternions for safety
    Q = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)

    return P3d, Q


def load_model_for_inference(model, ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model