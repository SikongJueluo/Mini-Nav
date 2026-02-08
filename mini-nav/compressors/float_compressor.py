import torch.nn as nn
import torch.nn.functional as F


class FloatCompressor(nn.Module):
    def __init__(self):
        super().__init__()

        # projection head
        self.proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )

        self.recover = nn.Linear(512, 1024)

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)  # [B,1024]

        z512 = self.proj(pooled)  # [B,512]
        z512 = F.normalize(z512, dim=-1)

        recon = self.recover(z512)  # [B,1024]

        return z512, recon
