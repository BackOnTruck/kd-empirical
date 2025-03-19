import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Callable

from params import *

def windowed(A: Tensor, W: int, f: Callable[[Tensor, Tensor], Tensor]):
    '''(B, L, N, H) => (B, L, N, 2W+1, H), pair-wise differences for L2 distance or products for cosine similarity'''
    N, H = A.shape[-2:]
    if W < N:
        index = torch.arange(N, device=A.device)[:, None] + torch.arange(-W, W + 1, device=A.device)[None, :] # (N, 2W+1)
        vec = f(A[:, :, :, None, :], A[:, :, index.clamp(min=0, max=N-1), :]) # (B, L, N, 2W+1, H)
        expanded = index[..., None].expand(-1, -1, H)
        vec *= (expanded >= 0) & (expanded < N)

    else:
        vec = f(A[:, :, :, None, :], A[:, :, None, :]) # (B, L, N, N, H)

    return vec

def cos_dist(A: Tensor, W: int):
    '''(B, L, N, H) => (B, L, N, 2W+1), pair-wise cosine distance of vectors with H elements'''
    A = F.normalize(A, p=2, dim=-1)
    return 1 - windowed(A, W, lambda x, y: x * y).sum(-1) # (B, L, N, 2W+1) or (B, L, N, N)

def dist_loss(A: Tensor, B: Tensor, W: int):
    return F.mse_loss(cos_dist(A, W), cos_dist(B, W))

def cos_angle(A: Tensor, W: int):
    '''(B, L, N, H) => (B, L, N, 2W+1, 2W+1), cosine of angles formed by 3 points on H-dimensional space'''
    vec = F.normalize(windowed(A, W, lambda x, y: x - y), p=2, dim=-1)
    return torch.einsum('blnwh,blnzh->blnwz', vec, vec) # (B, L, N, 2W+1, 2W+1) or (B, L, N, N, N)

def angle_loss(A: Tensor, B: Tensor, W: int):
    return F.mse_loss(cos_angle(A, W), cos_angle(B, W))

def ckd_loss(A: Tensor, B: Tensor, W: int):
    C, D = A.transpose(1, 2), B.transpose(1, 2)
    return dist_loss(A, B, W) + angle_loss(A, B, W) + (dist_loss(C, D, W) + angle_loss(C, D, W)) * GAMMA

if __name__ == "__main__":
    student = torch.randn(32, 6, 512, 512, device=DEVICE, requires_grad=True)
    teacher = torch.randn(32, 6, 512, 768, device=DEVICE, requires_grad=True)

    print(">>> Computing...")
    print(ckd_loss(student, teacher, 14))
