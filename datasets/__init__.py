"""
The `datasets` package provides modules for working with different datasets.

Submodules:
- `imagenet`: Module for working with the ImageNet dataset.
- `imagenette`: Module for working with the Imagenette dataset.
"""

from .imagenet import prepare_dataloaders
from .imagenette import prepare_dataloaders
