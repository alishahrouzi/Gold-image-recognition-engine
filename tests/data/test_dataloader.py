"""DataLoader smoke test for UnifiedDataset."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from data.collate import collate_samples
from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.datasets.unified_dataset import UnifiedDataset
from tests.data.helpers import write_manifest, write_rgb_image


def test_dataloader_compatibility(tmp_path: Path) -> None:
    rows = []
    for index, (group_id, category, split) in enumerate(
        [
            ("g1", "Bracelet", "train"),
            ("g1", "Bracelet", "train"),
            ("g2", "Ring", "train"),
            ("g3", "Necklace", "train"),
        ],
        start=1,
    ):
        image_path = write_rgb_image(tmp_path / f"{index}.jpg")
        rows.append(
            {
                "image_id": f"img_{index}",
                "image_path": str(image_path),
                "group_id": group_id,
                "category": category,
                "category_id": CATEGORY_TO_ID[category],
                "split": split,
                "source": SOURCE_DATASET1,
            }
        )

    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    dataset = UnifiedDataset(manifest, split="train")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_samples,
    )

    batch = next(iter(loader))
    assert len(batch["image"]) == 2
    assert len(batch["image_id"]) == 2
    assert len(batch["group_id"]) == 2
    assert len(batch["category"]) == 2
    assert len(batch["category_id"]) == 2
    assert len(batch["split"]) == 2
    assert len(batch["source"]) == 2
    assert batch["image"][0].mode == "RGB"
    assert all(source == SOURCE_DATASET1 for source in batch["source"])

    total = sum(len(item["image_id"]) for item in loader)
    assert total == 4
