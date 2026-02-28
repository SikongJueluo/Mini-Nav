"""Hash-based compressor for CAM-compatible binary codes.

Converts DINO features to 512-bit binary hash codes suitable for
Content Addressable Memory (CAM) retrieval.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common import BinarySign, hamming_similarity


class HashCompressor(nn.Module):
    """Compress DINO tokens to 512-bit binary codes for CAM storage.

    Architecture:
        tokens -> mean pool -> projection -> binary sign -> hash codes

    Output formats:
        - logits: continuous values for training (before sign)
        - hash_codes: {-1, +1} for similarity computation
        - bits: {0, 1} for CAM storage

    Example:
        >>> compressor = HashCompressor()
        >>> tokens = torch.randn(4, 197, 1024)  # DINO output
        >>> logits, hash_codes, bits = compressor(tokens)
        >>> bits.shape
        torch.Size([4, 512])
        >>> bits.dtype
        torch.int32
    """

    def __init__(self, input_dim: int = 1024, hash_bits: int = 512):
        """Initialize hash compressor.

        Args:
            input_dim: Input feature dimension (DINO output = 1024)
            hash_bits: Number of bits in hash code (CAM constraint = 512)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hash_bits = hash_bits

        # Projection head: maps DINO features to hash logits
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, hash_bits),
        )

        # Initialize last layer with smaller weights for stable training
        nn.init.xavier_uniform_(cast(Tensor, self.proj[-1].weight), gain=0.1)
        nn.init.zeros_(cast(Tensor, self.proj[-1].bias))

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass producing hash codes.

        Args:
            tokens: DINO patch tokens, shape [B, N, input_dim]

        Returns:
            Tuple of (logits, hash_codes, bits):
                - logits: [B, hash_bits] continuous values for training
                - hash_codes: [B, hash_bits] {-1, +1} values
                - bits: [B, hash_bits] {0, 1} values for CAM storage
        """
        # Pool tokens to single feature vector
        pooled = tokens.mean(dim=1)  # [B, input_dim]

        # Project to hash dimension
        logits = self.proj(pooled)  # [B, hash_bits]

        # Binary hash codes with STE for backprop
        hash_codes = BinarySign.apply(logits)  # [B, hash_bits] in {-1, +1}

        # Convert to bits for CAM storage
        bits = (hash_codes > 0).int()  # [B, hash_bits] in {0, 1}

        return logits, hash_codes, bits

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokens to binary bits for CAM storage.

        This is the inference-time method for database insertion.

        Args:
            tokens: DINO patch tokens, shape [B, N, input_dim]

        Returns:
            Binary bits [B, hash_bits] as int32 for CAM
        """
        _, _, bits = self.forward(tokens)
        return bits

    def compute_similarity(
        self, query_bits: torch.Tensor, db_bits: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamming similarity between query and database entries.

        Higher score = more similar (fewer differing bits).

        Args:
            query_bits: Query bits {0,1}, shape [Q, hash_bits]
            db_bits: Database bits {0,1}, shape [N, hash_bits]

        Returns:
            Similarity scores [Q, N], range [0, hash_bits]
        """
        # Convert bits to hash codes
        query_hash = query_bits * 2 - 1  # {0,1} -> {-1,+1}
        db_hash = db_bits * 2 - 1

        return hamming_similarity(query_hash, db_hash)


class HashLoss(nn.Module):
    """Batch-level retrieval loss for hash code learning.

    Combines three objectives:
        1. Contrastive: similar inputs have similar hash codes
        2. Distillation: hash preserves original DINO similarity structure
        3. Quantization: hash codes are close to binary {-1, +1}

    All losses are computed within batch - no full database retrieval needed.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        distill_weight: float = 0.5,
        quant_weight: float = 0.01,
        temperature: float = 0.2,
    ):
        """Initialize loss function.

        Args:
            contrastive_weight: Weight for contrastive loss
            distill_weight: Weight for distillation loss
            quant_weight: Weight for quantization loss
            temperature: Temperature for contrastive similarity scaling
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.distill_weight = distill_weight
        self.quant_weight = quant_weight
        self.temperature = temperature

    def contrastive_loss(
        self,
        logits: torch.Tensor,
        hash_codes: torch.Tensor,
        positive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """InfoNCE-style contrastive loss within batch.

        Learns that positive pairs (similar images) have similar hash codes,
        and negative pairs (different images) have dissimilar codes.

        Args:
            logits: Continuous logits [B, hash_bits]
            hash_codes: Binary hash codes {-1,+1} [B, hash_bits]
            positive_mask: Boolean mask [B, B] where True indicates positive pair
                          If None, uses identity matrix (each sample is its own positive)

        Returns:
            Scalar contrastive loss
        """
        batch_size = logits.size(0)
        device = logits.device

        # Use cosine similarity on continuous logits (more stable during training)
        logits_norm = F.normalize(logits, dim=-1)
        sim_matrix = logits_norm @ logits_norm.t() / self.temperature  # [B, B]

        # Create positive mask: diagonal is always positive (self-similarity)
        if positive_mask is None:
            positive_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        # InfoNCE: for each sample, positives should have high similarity
        # Mask out self-similarity for numerical stability
        mask_self = torch.eye(batch_size, device=device, dtype=torch.bool)
        sim_matrix_masked = sim_matrix.masked_fill(mask_self, float("-inf"))

        # For each anchor, positives are the target
        # We use a symmetric formulation: each positive pair contributes
        loss = 0.0
        num_positives = 0

        for i in range(batch_size):
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue

            # Numerator: similarity to positives
            pos_sim = sim_matrix[i, pos_indices]  # [num_positives]

            # Denominator: similarity to all negatives (including self as neg for stability)
            neg_sim = sim_matrix_masked[i]  # [B]

            # Log-sum-exp for numerical stability
            max_sim = neg_sim.max()
            log_denom = max_sim + torch.log(torch.exp(neg_sim - max_sim).sum())

            # Loss for this anchor
            loss += -pos_sim.mean() + log_denom
            num_positives += 1

        return loss / max(num_positives, 1)

    def distillation_loss(
        self,
        hash_codes: torch.Tensor,
        teacher_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Distillation loss preserving DINO similarity structure.

        Ensures that if two images are similar in DINO space,
        they remain similar in hash space.

        Args:
            hash_codes: Binary hash codes {-1,+1} [B, hash_bits]
            teacher_embed: DINO embeddings [B, teacher_dim], assumed normalized

        Returns:
            Scalar distillation loss
        """
        hash_bits = hash_codes.size(-1)

        # Hash similarity: inner product of {-1,+1} gives range [-hash_bits, hash_bits]
        hash_sim = hash_codes @ hash_codes.t()  # [B, B]
        hash_sim = hash_sim / hash_bits  # Normalize to [-1, 1]

        # Teacher similarity: cosine (assumes teacher_embed is normalized)
        teacher_sim = teacher_embed @ teacher_embed.t()  # [B, B]

        # MSE between similarity matrices
        loss = F.mse_loss(hash_sim, teacher_sim)

        return loss

    def quantization_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Quantization loss pushing logits toward {-1, +1}.

        Without this, logits stay near 0 and sign() is unstable.

        Args:
            logits: Continuous logits [B, hash_bits]

        Returns:
            Scalar quantization loss
        """
        # Push |logit| toward 1
        return torch.mean(torch.abs(logits.abs() - 1))

    def forward(
        self,
        logits: torch.Tensor,
        hash_codes: torch.Tensor,
        teacher_embed: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined hash training loss.

        Args:
            logits: Continuous logits [B, hash_bits]
            hash_codes: Binary hash codes {-1,+1} [B, hash_bits]
            teacher_embed: DINO embeddings [B, teacher_dim]
            positive_mask: Optional positive pair mask [B, B]

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Ensure teacher embeddings are normalized
        teacher_embed = F.normalize(teacher_embed, dim=-1)

        # Compute individual losses
        loss_cont = self.contrastive_loss(logits, hash_codes, positive_mask)
        loss_distill = self.distillation_loss(hash_codes, teacher_embed)
        loss_quant = self.quantization_loss(logits)

        # Combine
        total_loss = (
            self.contrastive_weight * loss_cont
            + self.distill_weight * loss_distill
            + self.quant_weight * loss_quant
        )

        # Return components for logging
        components = {
            "contrastive": loss_cont.item(),
            "distill": loss_distill.item(),
            "quantization": loss_quant.item(),
            "total": total_loss.item(),
        }

        return total_loss, components


class VideoPositiveMask:
    """Generate positive pair masks for video sequences.

    In indoor navigation, consecutive video frames are positive pairs
    (same location, different viewpoint/lighting).
    """

    def __init__(self, temporal_window: int = 3):
        """Initialize mask generator.

        Args:
            temporal_window: Frames within this distance are considered positive
        """
        self.temporal_window = temporal_window

    def from_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Create positive mask from frame indices.

        Args:
            frame_indices: Frame index for each sample [B]

        Returns:
            Boolean mask [B, B] where True indicates positive pair
        """
        batch_size = frame_indices.size(0)
        device = frame_indices.device

        # Compute temporal distance
        indices_i = frame_indices.unsqueeze(1)  # [B, 1]
        indices_j = frame_indices.unsqueeze(0)  # [1, B]
        temporal_dist = (indices_i - indices_j).abs()  # [B, B]

        # Positive if within temporal window
        positive_mask = temporal_dist <= self.temporal_window

        # Exclude self (diagonal will be handled separately in loss)
        # Actually keep it, loss handles self-similarity specially

        return positive_mask

    def from_video_ids(
        self, video_ids: torch.Tensor, frame_indices: torch.Tensor
    ) -> torch.Tensor:
        """Create positive mask considering both video ID and frame index.

        Args:
            video_ids: Video ID for each sample [B]
            frame_indices: Frame index within video [B]

        Returns:
            Boolean mask [B, B] where True indicates positive pair
        """
        batch_size = video_ids.size(0)
        device = video_ids.device

        # Same video
        same_video = video_ids.unsqueeze(1) == video_ids.unsqueeze(0)  # [B, B]

        # Temporal proximity
        temporal_dist = (frame_indices.unsqueeze(1) - frame_indices.unsqueeze(0)).abs()
        temporal_close = temporal_dist <= self.temporal_window

        # Positive if same video AND temporally close
        return same_video & temporal_close
