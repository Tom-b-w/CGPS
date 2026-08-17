import torch

from cgps.core import ConsensusAccumulator, refine_prompt_weights


def test_consensus_accumulator_filters_disagreement_and_low_confidence() -> None:
    accumulator = ConsensusAccumulator.create(2, 2, confidence_threshold=0.30)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    standard = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    candidate = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])

    accumulator.update(features, standard, candidate)

    assert accumulator.sample_count.tolist() == [2.0, 0.0]
    expected = torch.tensor([[2 / 5**0.5, 1 / 5**0.5], [0.0, 0.0]])
    torch.testing.assert_close(accumulator.centroids(), expected)


def test_prompt_refinement_returns_normalized_class_weights() -> None:
    accumulator = ConsensusAccumulator.create(2, 2)
    accumulator.feature_sum.copy_(torch.eye(2))
    accumulator.sample_count.fill_(1)
    standard = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    candidates = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    ]

    weights = refine_prompt_weights(
        torch.eye(2),
        torch.eye(2),
        accumulator,
        standard,
        candidates,
        selection_fraction=0.5,
        min_consensus_count=1,
    )

    torch.testing.assert_close(weights, torch.eye(2))
    torch.testing.assert_close(weights.norm(dim=0), torch.ones(2))
