# Implementation of 3D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 3D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 3D spatiotemporal positions (time, height, width).

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionGetter:
    """Generates and caches 3D spatiotemporal positions for patches in a grid.

    This class efficiently manages the generation of spatiotemporal coordinates for patches
    in a 3D grid (time, height, width), caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple, torch.Tensor] = {}

    def __call__(
        self, batch_size: int, time: int, view: int, scene: int, device: torch.device, time_list: list
    ) -> torch.Tensor:
        """Generates spatiotemporal positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            time: Number of temporal frames.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, time*height*width, 3) containing t,y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        cache_key = (time, view, scene, device, tuple(time_list))
        if cache_key not in self.position_cache:
            # 生成基础坐标
            # t_coords = torch.arange(1, time+1, device=device)  # 原代码 使用顺序时间 [1,2,3,...,time]
            t_coords = torch.tensor(time_list, device=device)  # 自定义时间坐标
            v_coords = torch.arange(1, view + 1, device=device)
            s_coords = torch.arange(1, scene + 1, device=device)

            # 生成所有组合 (t, v, s)
            positions = torch.cartesian_prod(t_coords, v_coords, s_coords)  # [t*v*s, 3]

            # 创建ego_status tokens: (t, 0, 0) for each time step
            # ego_status = torch.zeros(time, 3, device=device)
            # ego_status[:, 0] = torch.arange(1, time+1, device=device)

            # 修改为使用自定义时间坐标
            ego_status = torch.zeros(time, 3, device=device)
            ego_status[:, 0] = t_coords  # 直接使用自定义的t_coords

            # 重塑positions以便插入ego_status
            positions = positions.view(time, view * scene, 3)  # [t, v*s, 3]

            # 在每个时间步的view-scene组合前插入ego_status
            all_positions = []
            for t in range(time):
                all_positions.append(ego_status[t : t + 1])  # [1, 3]
                all_positions.append(positions[t])  # [v*s, 3]

            positions = torch.cat(all_positions, dim=0)  # [t*(v*s+1), 3]
            self.position_cache[cache_key] = positions

        cached_positions = self.position_cache[cache_key]
        return cached_positions.view(1, -1, 3).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding3D(nn.Module):
    """3D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    3D spatiotemporal positions (time, height, width). It handles the position-dependent
    rotation of features separately for temporal, vertical and horizontal dimensions.

    Args:
        temporal_frequency: Base frequency for temporal dimension. Default: 10.0 (for ~12 frames)
        spatial_frequency: Base frequency for spatial dimensions. Default: 100.0 (for ~16x32 grid)
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        temporal_frequency: Base frequency for computing temporal embeddings.
        spatial_frequency: Base frequency for computing spatial embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(
        self,
        temporal_frequency: float = 50.0,
        view_frequency: float = 10.0,
        spatial_frequency: float = 100.0,
        scaling_factor: float = 1.0,
    ):
        """Initializes the 3D RoPE module."""
        super().__init__()
        self.temporal_frequency = temporal_frequency
        self.view_frequency = view_frequency
        self.spatial_frequency = spatial_frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, base_freq: float, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            base_freq: Base frequency for this dimension.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, base_freq, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (base_freq**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 3D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 6 for 3D positioning.
            positions: Position tensor of shape (batch_size, n_tokens, 3) containing
                      the t, y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 3D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        # assert tokens.size(-1) % 3 == 0, "Feature dimension must be divisible by 3 for 3D RoPE"
        assert positions.ndim == 3 and positions.shape[-1] == 3, "Positions must have shape (batch_size, n_tokens, 3)"

        # Compute feature dimension for each spatiotemporal direction
        # feature_dim = tokens.size(-1) // 3
        temp_dim = tokens.size(-1) // 4
        view_dim = tokens.size(-1) // 4
        scene_dim = tokens.size(-1) // 2

        # Get frequency components for each dimension with different base frequencies
        max_temporal = int(positions[..., 0].max()) + 1
        max_view = int(positions[..., 1].max()) + 1
        max_scene = int(positions[..., 2].max()) + 1

        # Temporal dimension uses temporal_frequency
        cos_temp, sin_temp = self._compute_frequency_components(
            temp_dim, max_temporal, self.temporal_frequency, tokens.device, tokens.dtype
        )
        # Spatial dimensions use spatial_frequency
        cos_view, sin_view = self._compute_frequency_components(
            view_dim, max_view, self.view_frequency, tokens.device, tokens.dtype
        )
        cos_scene, sin_scene = self._compute_frequency_components(
            scene_dim, max_scene, self.spatial_frequency, tokens.device, tokens.dtype
        )

        # Split features for temporal, vertical and horizontal processing
        temporal_features, view_features, scene_features = tokens.split([temp_dim, view_dim, scene_dim], dim=-1)

        # Apply RoPE separately for each dimension with dimension-specific frequencies
        temporal_features = self._apply_1d_rope(temporal_features, positions[..., 0], cos_temp, sin_temp)
        view_features = self._apply_1d_rope(view_features, positions[..., 1], cos_view, sin_view)
        scene_features = self._apply_1d_rope(scene_features, positions[..., 2], cos_scene, sin_scene)

        # Combine processed features
        return torch.cat((temporal_features, view_features, scene_features), dim=-1)


class PositionGetter4D:
    """Generates and caches 4D spatiotemporal positions for patches in a grid.

    This class efficiently manages the generation of spatiotemporal coordinates for patches
    in a 4D grid (time, view, height, width), caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple, torch.Tensor] = {}

    def __call__(
        self, batch_size: int, time: int, views: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Generates spatiotemporal positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            time: Number of temporal frames.
            views: Number of camera views.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, time*views*height*width, 4) containing t,v,h,w coordinates
            for each position in the grid, repeated for each batch item.
        """
        cache_key = (time, views, height, width, device)
        if cache_key not in self.position_cache:
            t_coords = torch.arange(time, device=device)
            v_coords = torch.arange(views, device=device)
            h_coords = torch.arange(height, device=device)
            w_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(t_coords, v_coords, h_coords, w_coords)
            self.position_cache[cache_key] = positions

        cached_positions = self.position_cache[cache_key]
        return cached_positions.view(1, time * views * height * width, 4).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding4D(nn.Module):
    """4D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    4D spatiotemporal positions (time, view, height, width). It handles the position-dependent
    rotation of features separately for temporal, view, vertical and horizontal dimensions.

    Args:
        temporal_frequency: Base frequency for temporal dimension. Default: 10.0 (for ~12 frames)
        view_frequency: Base frequency for view dimension. Default: 10.0 (for ~3 views)
        spatial_frequency: Base frequency for spatial dimensions. Default: 100.0 (for ~16x16 grid)
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        temporal_frequency: Base frequency for computing temporal embeddings.
        view_frequency: Base frequency for computing view embeddings.
        spatial_frequency: Base frequency for computing spatial embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(
        self,
        temporal_frequency: float = 10.0,
        view_frequency: float = 10.0,
        spatial_frequency: float = 100.0,
        scaling_factor: float = 1.0,
    ):
        """Initializes the 4D RoPE module."""
        super().__init__()
        self.temporal_frequency = temporal_frequency
        self.view_frequency = view_frequency
        self.spatial_frequency = spatial_frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, base_freq: float, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            base_freq: Base frequency for this dimension.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, base_freq, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (base_freq**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 4D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4 for 4D positioning.
            positions: Position tensor of shape (batch_size, n_tokens, 4) containing
                      the t, v, h and w coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 4D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 4 == 0, "Feature dimension must be divisible by 4 for 4D RoPE"
        assert positions.ndim == 3 and positions.shape[-1] == 4, "Positions must have shape (batch_size, n_tokens, 4)"

        # Compute feature dimension for each dimension
        feature_dim = tokens.size(-1) // 4

        # Get frequency components for each dimension with different base frequencies
        max_temporal = int(positions[..., 0].max()) + 1
        max_view = int(positions[..., 1].max()) + 1
        max_vertical = int(positions[..., 2].max()) + 1
        max_horizontal = int(positions[..., 3].max()) + 1

        # Temporal dimension uses temporal_frequency
        cos_temp, sin_temp = self._compute_frequency_components(
            feature_dim, max_temporal, self.temporal_frequency, tokens.device, tokens.dtype
        )
        # View dimension uses view_frequency
        cos_view, sin_view = self._compute_frequency_components(
            feature_dim, max_view, self.view_frequency, tokens.device, tokens.dtype
        )
        # Spatial dimensions use spatial_frequency
        cos_vert, sin_vert = self._compute_frequency_components(
            feature_dim, max_vertical, self.spatial_frequency, tokens.device, tokens.dtype
        )
        cos_hori, sin_hori = self._compute_frequency_components(
            feature_dim, max_horizontal, self.spatial_frequency, tokens.device, tokens.dtype
        )

        # Split features for temporal, view, vertical and horizontal processing
        temporal_features, view_features, vertical_features, horizontal_features = tokens.chunk(4, dim=-1)

        # Apply RoPE separately for each dimension with dimension-specific frequencies
        temporal_features = self._apply_1d_rope(temporal_features, positions[..., 0], cos_temp, sin_temp)
        view_features = self._apply_1d_rope(view_features, positions[..., 1], cos_view, sin_view)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 2], cos_vert, sin_vert)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 3], cos_hori, sin_hori)

        # Combine processed features
        return torch.cat((temporal_features, view_features, vertical_features, horizontal_features), dim=-1)
