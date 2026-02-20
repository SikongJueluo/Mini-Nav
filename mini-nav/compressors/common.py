"""Common utilities for compressor modules."""

import torch
import torch.nn.functional as F


class BinarySign(torch.autograd.Function):
    """Binary sign function with Straight-Through Estimator (STE).

    Forward: returns sign(x) in {-1, +1}
    Backward: passes gradients through as if identity

    For CAM storage, convert: bits = (sign_output + 1) / 2
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # STE: treat as identity
        # Optional: gradient clipping for stability
        return grad_output.clone()


def hamming_distance(b1, b2):
    """Compute Hamming distance between binary codes.

    Args:
        b1: Binary codes {0,1}, shape [N, D] or [D]
        b2: Binary codes {0,1}, shape [M, D] or [D]

    Returns:
        Hamming distances, shape [N, M] or scalar
    """
    if b1.dim() == 1 and b2.dim() == 1:
        return (b1 != b2).sum()

    # Expand for pairwise computation
    b1 = b1.unsqueeze(1)  # [N, 1, D]
    b2 = b2.unsqueeze(0)  # [1, M, D]

    return (b1 != b2).sum(dim=-1)  # [N, M]


def hamming_similarity(h1, h2):
    """Compute Hamming similarity for {-1, +1} codes.

    Args:
        h1: Hash codes {-1, +1}, shape [N, D] or [D]
        h2: Hash codes {-1, +1}, shape [M, D] or [D]

    Returns:
        Similarity scores in [-D, D], shape [N, M] or scalar
        Higher is more similar
    """
    if h1.dim() == 1 and h2.dim() == 1:
        return (h1 * h2).sum()

    return h1 @ h2.t()  # [N, M]


def bits_to_hash(b):
    """Convert {0,1} bits to {-1,+1} hash codes.

    Args:
        b: Binary bits {0,1}, any shape

    Returns:
        Hash codes {-1,+1}, same shape
    """
    return b * 2 - 1


def hash_to_bits(h):
    """Convert {-1,+1} hash codes to {0,1} bits.

    Args:
        h: Hash codes {-1,+1}, any shape

    Returns:
        Binary bits {0,1}, same shape
    """
    return (h + 1) / 2
