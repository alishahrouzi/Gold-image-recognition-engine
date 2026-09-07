import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from baselines.classifier import BaselineClassifier
from baselines.config import BaselineConfig
from baselines.metrics import classification_metrics
from baselines.objective import classification_step


class _ToyRetrieval(nn.Module):
    embedding_dim = 4

    def forward(self, images):
        return nn.functional.normalize(images, dim=1)


class _ToyDataset(Dataset):
    def __init__(self):
        self.images = torch.eye(4)
        self.labels = [0, 1, 2, 3]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"image": self.images[index], "category_id": self.labels[index]}


def _collate(items):
    return {
        "image": torch.stack([item["image"] for item in items]),
        "category_id": [item["category_id"] for item in items],
    }


def test_baseline_config_maps_to_training_contract():
    cfg = BaselineConfig()
    training = cfg.to_training_config()
    assert training.embedding_dim == 128
    assert training.loss_name == "contrastive"
    assert cfg.as_loggable_dict()["objective"] == "category_cross_entropy"


def test_classification_step_returns_finite_scalar():
    retrieval = _ToyRetrieval()
    model = BaselineClassifier(retrieval, num_classes=4)
    batch = {"image": torch.eye(4), "category_id": [0, 1, 2, 3]}
    loss = classification_step(model, batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_classification_metrics_reports_accuracy_and_embedding_norm():
    retrieval = _ToyRetrieval()
    model = BaselineClassifier(retrieval, num_classes=4)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.eye(4))
        model.classifier.bias.zero_()
    loader = DataLoader(_ToyDataset(), batch_size=2, collate_fn=_collate)
    metrics = classification_metrics(model, loader)
    assert metrics["accuracy"] == 1.0
    assert abs(metrics["embedding_norm_mean"] - 1.0) < 1e-6
