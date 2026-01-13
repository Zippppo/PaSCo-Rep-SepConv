import torch
import torch.nn.functional as F
import numpy as np


def sinusoidal_positional_encoding(coords, num_freqs=5, pad_to=32):
    """
    Sinusoidal positional encoding for 3D coordinates.

    Args:
        coords: [N, 3] tensor of coordinates normalized to [0, 1]
        num_freqs: number of frequency bands (default 5)
        pad_to: pad output to this dimension (default 32)

    Returns:
        [N, pad_to] tensor of positional encodings

    Formula:
        For each coordinate dimension:
        [sin(2^0 * pi * x), cos(2^0 * pi * x),
         sin(2^1 * pi * x), cos(2^1 * pi * x),
         ...,
         sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]

        Total: 3 dims * num_freqs * 2 (sin/cos) = 30D for num_freqs=5
    """
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).float()

    device = coords.device
    N = coords.shape[0]

    # Frequency bands: [1, 2, 4, 8, 16] for num_freqs=5
    freqs = 2.0 ** torch.arange(num_freqs, device=device, dtype=coords.dtype)

    # Expand coords for broadcasting: [N, 3, 1] * [1, 1, num_freqs] -> [N, 3, num_freqs]
    coords_expanded = coords.unsqueeze(-1)  # [N, 3, 1]
    freqs_expanded = freqs.view(1, 1, -1)   # [1, 1, num_freqs]

    # Compute angles: [N, 3, num_freqs]
    angles = coords_expanded * freqs_expanded * np.pi

    # Sin and cos encodings: each [N, 3, num_freqs]
    sin_enc = torch.sin(angles)
    cos_enc = torch.cos(angles)

    # Interleave sin and cos: [N, 3, num_freqs * 2]
    # Order: [sin_f0, cos_f0, sin_f1, cos_f1, ...]
    encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # [N, 3, num_freqs, 2]
    encoding = encoding.view(N, 3, -1)  # [N, 3, num_freqs * 2]

    # Flatten to [N, 3 * num_freqs * 2] = [N, 30] for num_freqs=5
    encoding = encoding.view(N, -1)

    # Pad to target dimension if needed
    current_dim = encoding.shape[-1]
    if current_dim < pad_to:
        encoding = F.pad(encoding, (0, pad_to - current_dim))

    return encoding


def normalize_coordinates(coords, grid_min, grid_max):
    """
    Normalize coordinates from world space to [0, 1] range.

    Args:
        coords: [N, 3] coordinates in world space (mm)
        grid_min: [3] minimum bounds
        grid_max: [3] maximum bounds

    Returns:
        [N, 3] normalized coordinates in [0, 1]
    """
    if isinstance(coords, np.ndarray):
        grid_min = np.asarray(grid_min)
        grid_max = np.asarray(grid_max)
        extent = grid_max - grid_min
        extent = np.where(extent > 0, extent, 1.0)  # avoid division by zero
        return (coords - grid_min) / extent
    else:
        grid_min = torch.as_tensor(grid_min, dtype=coords.dtype, device=coords.device)
        grid_max = torch.as_tensor(grid_max, dtype=coords.dtype, device=coords.device)
        extent = grid_max - grid_min
        extent = torch.where(extent > 0, extent, torch.ones_like(extent))
        return (coords - grid_min) / extent
