"""Loaders for manifests and RGB images."""

from .image_loader import load_rgb_image
from .manifest import load_manifest

__all__ = ["load_manifest", "load_rgb_image"]
