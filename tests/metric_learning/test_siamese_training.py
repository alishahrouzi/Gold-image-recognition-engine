from pathlib import Path

import torch
from PIL import Image

from data.pairs.dataset import PairDataset
from data.pairs.types import PAIR_TYPE_NEGATIVE, PAIR_TYPE_POSITIVE, Pair, make_pair_id
from data.types import Sample
from metric_learning.distance import EuclideanDistance
from metric_learning.losses import ContrastiveLoss
from metric_learning.siamese import SiameseNetwork
from metric_learning.training import contrastive_step, make_contrastive_step


def _sample(tmp_path: Path, image_id: str, group_id: str, category: str, category_id: int) -> Sample:
    path = tmp_path / f"{image_id}.jpg"
    Image.new("RGB", (32, 32), (category_id * 20 + 20, 40, 60)).save(path)
    return Sample(
        image_id=image_id,
        image_path=path,
        group_id=group_id,
        category=category,
        category_id=category_id,
        split="train",
        source="dataset1",
    )


def _pair(a: Sample, b: Sample, *, positive: bool) -> Pair:
    return Pair(
        pair_id=make_pair_id(a.image_id, b.image_id),
        image_id_1=min(a.image_id, b.image_id),
        image_id_2=max(a.image_id, b.image_id),
        group_id_1=a.group_id if a.image_id < b.image_id else b.group_id,
        group_id_2=b.group_id if a.image_id < b.image_id else a.group_id,
        category_id_1=a.category_id if a.image_id < b.image_id else b.category_id,
        category_id_2=b.category_id if a.image_id < b.image_id else a.category_id,
        category_1=a.category if a.image_id < b.image_id else b.category,
        category_2=b.category if a.image_id < b.image_id else a.category,
        split="train",
        label=1 if positive else 0,
        pair_type=PAIR_TYPE_POSITIVE if positive else PAIR_TYPE_NEGATIVE,
        negative_type=None if positive else "same_category",
    )


def test_pair_dataset_converts_s1_10_label_for_contrastive_loss(tmp_path: Path) -> None:
    a = _sample(tmp_path, "a", "g1", "Bracelet", 0)
    b = _sample(tmp_path, "b", "g1", "Bracelet", 0)
    c = _sample(tmp_path, "c", "g2", "Bracelet", 0)
    dataset = PairDataset([_pair(a, b, positive=True), _pair(a, c, positive=False)], [a, b, c])

    positive = dataset[0]
    negative = dataset[1]

    assert positive["image_a"].shape == (3, 224, 224)
    assert positive["image_b"].shape == (3, 224, 224)
    assert positive["label"].item() == 0.0
    assert positive["pair_label"].item() == 1
    assert negative["label"].item() == 1.0
    assert negative["pair_label"].item() == 0


def test_contrastive_step_uses_distance_and_loss_contract() -> None:
    torch.manual_seed(7)
    model = SiameseNetwork()
    image_a = torch.randn(2, 3, 224, 224)
    image_b = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0.0, 1.0])
    batch = {"image_a": image_a, "image_b": image_b, "label": labels}
    loss_fn = ContrastiveLoss(margin=1.0)
    distance = EuclideanDistance()

    loss = contrastive_step(model, batch, distance=distance, loss_fn=loss_fn)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_make_contrastive_step_is_trainer_compatible() -> None:
    step = make_contrastive_step(
        distance=EuclideanDistance(),
        loss_fn=ContrastiveLoss(margin=1.0),
    )
    assert callable(step)
