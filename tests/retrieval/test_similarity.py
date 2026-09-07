import pytest
import torch

from src.retrieval.similarity import cosine_similarity_matrix


def test_cosine_similarity_matrix_matches_expected_values() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    scores = cosine_similarity_matrix(query, gallery)

    assert scores.shape == (2, 2)
    assert scores[0, 0].item() == pytest.approx(1.0)
    assert scores[0, 1].item() == pytest.approx(2 ** -0.5)
    assert scores[1, 0].item() == pytest.approx(0.0)
    assert scores[1, 1].item() == pytest.approx(2 ** -0.5)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity_matrix(torch.ones(1, 3), torch.ones(2, 4))
