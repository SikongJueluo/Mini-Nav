"""Feature compression module with attention-based pooling and MLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolNetCompressor(nn.Module):
    """Pool + Network feature compressor for DINOv2 embeddings.

    Combines attention-based Top-K token pooling with a 2-layer MLP to compress
    DINOv2's last_hidden_state from [batch, seq_len, hidden_dim] to [batch, compression_dim].

    Args:
        input_dim: Input feature dimension (e.g., 1024 for DINOv2-large)
        compression_dim: Output feature dimension (default: 256)
        top_k_ratio: Ratio of tokens to keep via attention pooling (default: 0.5)
        hidden_ratio: Hidden layer dimension as multiple of compression_dim (default: 2.0)
        dropout_rate: Dropout probability (default: 0.1)
        use_residual: Whether to use residual connection (default: True)
        device: Device to place model on ('auto', 'cpu', or 'cuda')
    """

    def __init__(
        self,
        input_dim: int,
        compression_dim: int = 256,
        top_k_ratio: float = 0.5,
        hidden_ratio: float = 2.0,
        dropout_rate: float = 0.1,
        use_residual: bool = True,
        device: str = "auto",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.compression_dim = compression_dim
        self.top_k_ratio = top_k_ratio
        self.use_residual = use_residual

        # Attention mechanism for token selection
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
        )

        # Compression network: 2-layer MLP
        hidden_dim = int(compression_dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, compression_dim),
        )

        # Residual projection if dimensions don't match
        if use_residual and input_dim != compression_dim:
            self.residual_proj = nn.Linear(input_dim, compression_dim)
        else:
            self.residual_proj = None

        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def _compute_attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention scores for each token.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Attention scores [batch, seq_len, 1]
        """
        scores = self.attention(x)  # [batch, seq_len, 1]
        return scores.squeeze(-1)  # [batch, seq_len]

    def _apply_pooling(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply Top-K attention pooling to select important tokens.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            scores: Attention scores [batch, seq_len]

        Returns:
            Pooled features [batch, k, input_dim] where k = ceil(seq_len * top_k_ratio)
        """
        batch_size, seq_len, _ = x.shape
        k = max(1, int(seq_len * self.top_k_ratio))

        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(scores, k=k, dim=-1)  # [batch, k]

        # Select top-k tokens
        batch_indices = (
            torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)
        )
        pooled = x[batch_indices, top_k_indices, :]  # [batch, k, input_dim]

        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through compressor.

        Args:
            x: Input features [batch, seq_len, input_dim]

        Returns:
            Compressed features [batch, compression_dim]
        """
        # Compute attention scores
        scores = self._compute_attention_scores(x)

        # Apply Top-K pooling
        pooled = self._apply_pooling(x, scores)

        # Average pool over selected tokens to get [batch, input_dim]
        pooled = pooled.mean(dim=1)  # [batch, input_dim]

        # Apply compression network
        compressed = self.net(pooled)  # [batch, compression_dim]

        # Add residual connection if enabled
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(pooled)
            else:
                residual = pooled[:, : self.compression_dim]
            compressed = compressed + residual

        return compressed
