"""
The `utils` package provides utility functions for various tasks, including image processing and system operations.

Submodules:
- `images`: Utility functions for image processing.
- `system`: Utility functions for system operations.
"""

from .images import convert_to_rgb, normalize_image, get_attention_maps, plot_attention_maps
from .system import calculate_num_workers
