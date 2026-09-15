"""S4.7 helpers for resolving gallery product images for the MVP UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

from inference.gallery import RuntimeGallery


class ProductImageResolver:
    """Resolve one representative catalog image from trusted gallery metadata."""

    def __init__(self, gallery: RuntimeGallery) -> None:
        self._gallery = gallery

    def resolve(self, product_id: str) -> Path:
        """Return the first existing image belonging to ``product_id``."""
        target = str(product_id).strip()
        if not target:
            raise HTTPException(status_code=404, detail="Product image not found.")

        for item in self._gallery.gallery.metadata:
            item_product = str(item.get("product_id", item.get("product_group", ""))).strip()
            if item_product != target:
                continue
            raw_path = item.get("image", item.get("image_path", ""))
            path = Path(str(raw_path))
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                return path

        raise HTTPException(status_code=404, detail="Product image not found.")

    def response(self, product_id: str) -> FileResponse:
        """Return the representative product image as an HTTP response."""
        path = self.resolve(product_id)
        return FileResponse(path)
