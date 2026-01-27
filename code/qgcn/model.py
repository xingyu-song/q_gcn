import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel recalibration)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, V, T)  OR (B, C, E, T)
        """
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))              # (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))      # (B, C)
        s = s.view(b, c, 1, 1)
        return x * s


def _normalize_graph(A: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Paper uses normalized adjacency: Abar = Lambda^{-1/2} A Lambda^{-1/2} with alpha=0.001
    (see Eq.4/5 description).  [oai_citation:4‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
    A: (..., N, N) or (..., B, B)
    """
    # degree = sum over last dim
    deg = A.sum(dim=-1) + eps
    deg_inv_sqrt = deg.rsqrt()
    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


# ----------------------------
# Directed Spatial Graph Conv
# ----------------------------

@dataclass
class DirectedGraphMatrices:
    """
    Provide the matrices needed for directed vertex/edge conv.

    For vertex conv (producing vertex features):
      - Av_k : adjacency (vertex->vertex) for subset k, shape (K, N, N)
      - Pe_k : parent incidence (edge->vertex) subset k, shape (K, N, B)
      - Ce_k : child incidence  (edge->vertex) subset k, shape (K, N, B)

    For edge conv (producing edge features):
      - Ae_k : adjacency (edge->edge) for subset k, shape (K, B, B)
      - Pv_k : parent incidence (vertex->edge) subset k, shape (K, B, N)
      - Cv_k : child incidence  (vertex->edge) subset k, shape (K, B, N)

    Q-GCN uses K=4 subsets for both vertices and edges.  [oai_citation:5‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
    """
    Av: torch.Tensor  # (K, N, N)
    Pe: torch.Tensor  # (K, N, B)
    Ce: torch.Tensor  # (K, N, B)

    Ae: torch.Tensor  # (K, B, B)
    Pv: torch.Tensor  # (K, B, N)
    Cv: torch.Tensor  # (K, B, N)


class DirectedSpatialConv(nn.Module):
    """
    Spatial graph conv that matches the paper’s "adaptive representation"
    idea: sum over K subsets, with concatenated matrices [Abar, Pbar, Cbar]
    then a learnable 1x1 conv (implemented as Linear over channels).  [oai_citation:6‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)

    We implement it in a practical, readable way:
      - aggregate neighbors with matrices
      - apply per-subset linear projection
      - sum over subsets

    Input layout:
      x_v: (B, C_in, N, T)
      x_e: (B, C_in, B, T)
    """
    def __init__(self, c_in: int, c_out: int, K: int, mode: str):
        super().__init__()
        assert mode in {"vertex", "edge"}
        self.mode = mode
        self.K = K
        # One projection per subset (acts like the 1x1 conv weight W_k in Eq.4/5)
        self.proj = nn.ModuleList([nn.Conv2d(c_in, c_out, kernel_size=1) for _ in range(K)])

    def forward(
        self,
        x: torch.Tensor,
        mats: DirectedGraphMatrices,
        x_other: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        If mode == "vertex":
          x is vertex features (B,C,N,T)
          x_other is edge features   (B,C,B,T)  (needed for incidence aggregation)

        If mode == "edge":
          x is edge features (B,C,B,T)
          x_other is vertex features (B,C,N,T)
        """
        if self.mode == "vertex":
            assert x_other is not None, "vertex conv needs edge features for incidence aggregation"
            # matrices: Av (K,N,N), Pe/Ce (K,N,B)
            Av = _normalize_graph(mats.Av)     # (K,N,N)
            Pe = mats.Pe
            Ce = mats.Ce
            # aggregate per subset
            out = 0.0
            for k in range(self.K):
                # neighbor vertex aggregation: (N,N) @ (B,C,N,T) -> (B,C,N,T)
                xv = torch.einsum("ij,bcjt->bcit", Av[k], x)
                # incidence from edges: (N,B) @ (B_batch,C_channels,B_bones,T_time) -> (B_batch,C_channels,N,T_time)
                xp = torch.einsum("ne,bcet->bcnt", Pe[k], x_other)
                xc = torch.einsum("ne,bcet->bcnt", Ce[k], x_other)
                agg = xv + xp + xc
                out = out + self.proj[k](agg)
            return out

        else:
            assert x_other is not None, "edge conv needs vertex features for incidence aggregation"
            # matrices: Ae (K,B,B), Pv/Cv (K,B,N)
            Ae = _normalize_graph(mats.Ae)     # (K,B,B)
            Pv = mats.Pv
            Cv = mats.Cv
            out = 0.0
            for k in range(self.K):
                xe = torch.einsum("ij,bcjt->bcit", Ae[k], x)          # (B,C,B,T)
                xp = torch.einsum("in,bcnt->bcit", Pv[k], x_other)    # (B,C,B,T)
                xc = torch.einsum("in,bcnt->bcit", Cv[k], x_other)    # (B,C,B,T)
                agg = xe + xp + xc
                out = out + self.proj[k](agg)
            return out


# ----------------------------
# Spatio-temporal Block (Spatial graph conv + temporal conv)
# ----------------------------

class DirectedSTGCNBlock(nn.Module):
    """
    One "convolutional block" in Q-GCN:
      spatial (directed graph conv) -> temporal conv (1 x T_k) -> BN -> ReLU -> Dropout
      with residual connection.  [oai_citation:7‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        K: int,
        mode: str,                 # "vertex" or "edge"
        t_kernel: int = 9,
        t_stride: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.mode = mode
        self.spatial = DirectedSpatialConv(c_in, c_out, K=K, mode=mode)

        # 1 x T temporal conv (over time dimension), keeping joint/edge dim as "height"
        pad = (t_kernel - 1) // 2
        self.temporal = nn.Conv2d(
            c_out, c_out,
            kernel_size=(1, t_kernel),
            stride=(1, t_stride),
            padding=(0, pad),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout(dropout)

        # Residual path (match channels/stride if needed)
        if (c_in == c_out) and (t_stride == 1):
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=(1, t_stride), bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(
        self,
        x: torch.Tensor,
        mats: DirectedGraphMatrices,
        x_other: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:      (B, C_in, V, T) or (B, C_in, E, T)
        x_other:(B, C_in, E, T) or (B, C_in, V, T)
        """
        res = self.res(x)
        y = self.spatial(x, mats=mats, x_other=x_other)
        y = self.temporal(y)
        y = self.bn(y)
        y = F.relu(y + res, inplace=True)
        y = self.drop(y)
        return y


# ----------------------------
# Q-GCN Model (Architecture)
# ----------------------------

class QGCN(nn.Module):
    """
    High-level Q-GCN:
      Inputs:
        - 2D node coordinates P: (B, T, N, 2)   [oai_citation:8‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
        - 2D bone rotations  R: (B, T, Bn, 2)  (cos,sin)  [oai_citation:9‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
      Outputs:
        - 3D joint positions: (B, T', N, 3)
        - 4D bone quaternions: (B, T', Bn, 4)

    Notes:
      - The paper diagram shows feature channels 16 -> 32 -> 64 and temporal downsample T/2, T/4, T/8.  [oai_citation:10‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
      - Uses SE blocks before FC heads.  [oai_citation:11‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
    """
    def __init__(
        self,
        num_joints: int,
        num_bones: int,
        mats: DirectedGraphMatrices,
        K: int = 4,
        t_kernel: int = 9,
        dropout: float = 0.25,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.N = num_joints
        self.B = num_bones
        self.K = K

        # Register matrices as buffers (so they move with .to(device))
        self.register_buffer("Av", mats.Av)
        self.register_buffer("Pe", mats.Pe)
        self.register_buffer("Ce", mats.Ce)
        self.register_buffer("Ae", mats.Ae)
        self.register_buffer("Pv", mats.Pv)
        self.register_buffer("Cv", mats.Cv)

        # Input projections (2 -> C)
        self.v_in = nn.Conv2d(2, 16, kernel_size=1)
        self.e_in = nn.Conv2d(2, 16, kernel_size=1)

        # Feature extraction stages (roughly matching Fig.1 block count/order)  [oai_citation:12‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
        # Stage 1: 16ch, stride 2 (T -> T/2)
        self.v1 = DirectedSTGCNBlock(16, 16, K=K, mode="vertex", t_kernel=t_kernel, t_stride=2, dropout=dropout)
        self.e1 = DirectedSTGCNBlock(16, 16, K=K, mode="edge",   t_kernel=t_kernel, t_stride=2, dropout=dropout)

        # Stage 2: 32ch, stride 2 (T/2 -> T/4)
        self.v2 = DirectedSTGCNBlock(16, 32, K=K, mode="vertex", t_kernel=t_kernel, t_stride=2, dropout=dropout)
        self.e2 = DirectedSTGCNBlock(16, 32, K=K, mode="edge",   t_kernel=t_kernel, t_stride=2, dropout=dropout)

        # Stage 3: 64ch, stride 2 (T/4 -> T/8)
        self.v3 = DirectedSTGCNBlock(32, 64, K=K, mode="vertex", t_kernel=t_kernel, t_stride=2, dropout=dropout)
        self.e3 = DirectedSTGCNBlock(32, 64, K=K, mode="edge",   t_kernel=t_kernel, t_stride=2, dropout=dropout)

        # SE blocks for both streams (pose & rotation features)  [oai_citation:13‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
        self.se_v = SEBlock(64, reduction=se_reduction)
        self.se_e = SEBlock(64, reduction=se_reduction)

        # Pose head: 64 -> 3 per joint (per time step)
        # (Paper says FC integrates multi-scale feature maps; this is a clean baseline head.)  [oai_citation:14‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
        self.pose_fc = nn.Conv2d(64, 3, kernel_size=1)

        # Orientation head:
        # concatenate rotation features with predicted 3D coordinates, then 2 FC layers with BN+ReLU in between  [oai_citation:15‡2404.19279v3.pdf](sediment://file_00000000f98c7206832b135f068d032c)
        self.ori_fc1 = nn.Conv2d(64 + 3, 128, kernel_size=1)
        self.ori_bn1 = nn.BatchNorm2d(128)
        self.ori_fc2 = nn.Conv2d(128, 4, kernel_size=1)

    def _mats(self) -> DirectedGraphMatrices:
        return DirectedGraphMatrices(
            Av=self.Av, Pe=self.Pe, Ce=self.Ce,
            Ae=self.Ae, Pv=self.Pv, Cv=self.Cv
        )

    def forward(self, P_2d: torch.Tensor, R_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        P_2d: (B, T, N, 2)
        R_2d: (B, T, Bn, 2)   (cos,sin) rotations per bone in 2D
        Returns:
          P_3d: (B, T', N, 3)
          Q_4d: (B, T', Bn, 4)
        """
        B, T, N, _ = P_2d.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"
        assert R_2d.shape[0] == B and R_2d.shape[1] == T and R_2d.shape[3] == 2
        assert R_2d.shape[2] == self.B, f"Expected num_bones={self.B}, got {R_2d.shape[2]}"

        mats = self._mats()

        # Convert to (B, C, V, T)
        xv_init = P_2d.permute(0, 3, 2, 1).contiguous()   # (B,2,N,T)
        xe_init = R_2d.permute(0, 3, 2, 1).contiguous()   # (B,2,Bn,T)

        xv = self.v_in(xv_init)                           # (B,16,N,T)
        xe = self.e_in(xe_init)                           # (B,16,Bn,T)

        # Stage 1: Inputs are (B,16,N,T) and (B,16,Bn,T). Outputs (T/2)
        # Store inputs for this stage to be used as x_other for cross-stream connections
        xv_input_stage1 = xv
        xe_input_stage1 = xe

        xv_out_stage1 = self.v1(xv_input_stage1, mats=mats, x_other=xe_input_stage1)      # (B,16,N,T/2)
        xe_out_stage1 = self.e1(xe_input_stage1, mats=mats, x_other=xv_input_stage1)      # (B,16,Bn,T/2)

        xv = xv_out_stage1
        xe = xe_out_stage1

        # Stage 2: Inputs are (B,16,N,T/2) and (B,16,Bn,T/2). Outputs (T/4)
        xv_input_stage2 = xv
        xe_input_stage2 = xe

        xv_out_stage2 = self.v2(xv_input_stage2, mats=mats, x_other=xe_input_stage2)      # (B,32,N,T/4)
        xe_out_stage2 = self.e2(xe_input_stage2, mats=mats, x_other=xv_input_stage2)      # (B,32,Bn,T/4)

        xv = xv_out_stage2
        xe = xe_out_stage2

        # Stage 3: Inputs are (B,32,N,T/4) and (B,32,Bn,T/4). Outputs (T/8)
        xv_input_stage3 = xv
        xe_input_stage3 = xe

        xv_out_stage3 = self.v3(xv_input_stage3, mats=mats, x_other=xe_input_stage3)      # (B,64,N,T/8)
        xe_out_stage3 = self.e3(xe_input_stage3, mats=mats, x_other=xv_input_stage3)      # (B,64,Bn,T/8)

        xv = xv_out_stage3
        xe = xe_out_stage3

        # SE
        xv = self.se_v(xv)
        xe = self.se_e(xe)

        # Pose prediction: (B,3,N,T')
        p3 = self.pose_fc(xv)

        # Orientation prediction:
        # concat edge features with predicted 3D coordinates "per bone".
        # If you have a bone->(parent,child) mapping, you can build bone-wise coords.
        # Here we do a simple, common baseline: use parent joint coord for each bone (must be provided by a mapping).
        # For pure architecture, we keep it generic: expect user to swap in their own bone-wise 3D features.
        #
        # We'll create a placeholder bone-wise 3D tensor by averaging all joints (NOT for training—just shape-safe).
        p3_mean = p3.mean(dim=2, keepdim=True)                   # (B,3,1,T')
        p3_bone = p3_mean.expand(-1, -1, self.B, -1).contiguous()# (B,3,Bn,T')

        ori_in = torch.cat([xe, p3_bone], dim=1)                 # (B,64+3,Bn,T')
        q = self.ori_fc1(ori_in)
        q = self.ori_bn1(q)
        q = F.relu(q, inplace=True)
        q = self.ori_fc2(q)                                      # (B,4,Bn,T')

        # Back to (B, T', N, 3) and (B, T', Bn, 4)
        P_3d = p3.permute(0, 3, 2, 1).contiguous()
        Q_4d = q.permute(0, 3, 2, 1).contiguous()

        return P_3d, Q_4d




