from typing import Optional, cast

import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, Dinov2Model


class DinoCompressor(nn.Module):
    """DINOv2 feature extractor with optional hash compression.

    When compressor is None: returns normalized DINO embeddings.
    When compressor is provided: returns binary hash bits for CAM storage.
    """

    def __init__(self, compressor: Optional[nn.Module] = None):
        super().__init__()

        self.dino = cast(
            Dinov2Model,
            AutoModel.from_pretrained("facebook/dinov2-large"),
        )

        self.compressor = compressor

    def forward(self, inputs):
        teacher_tokens = self.dino(**inputs).last_hidden_state  # [B,N,1024]

        teacher_embed = teacher_tokens.mean(dim=1)
        teacher_embed = F.normalize(teacher_embed, dim=-1)  # [B,1024]

        if self.compressor is None:
            return teacher_embed

        # HashCompressor returns (logits, hash_codes, bits)
        _, _, bits = self.compressor(teacher_tokens)
        return bits  # [B, 512] binary bits for CAM
