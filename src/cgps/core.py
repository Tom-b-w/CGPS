"""Testable extraction of the CGA and VGPS algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass
class ConsensusAccumulator:
    """Accumulate class centroids only from confident classifier agreement."""

    feature_sum: torch.Tensor
    sample_count: torch.Tensor
    confidence_threshold: float

    @classmethod
    def create(
        cls,
        num_classes: int,
        feature_dim: int,
        device: torch.device | str = "cpu",
        confidence_threshold: float = 0.30,
    ) -> ConsensusAccumulator:
        if num_classes < 1 or feature_dim < 1:
            raise ValueError("num_classes and feature_dim must be positive.")
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1].")
        return cls(
            feature_sum=torch.zeros(num_classes, feature_dim, device=device),
            sample_count=torch.zeros(num_classes, device=device),
            confidence_threshold=confidence_threshold,
        )

    @torch.no_grad()
    def update(
        self,
        image_features: torch.Tensor,
        standard_probabilities: torch.Tensor,
        candidate_probabilities: torch.Tensor,
    ) -> None:
        """Update state for samples on which both classifiers agree confidently."""

        if image_features.ndim != 2:
            raise ValueError("image_features must have shape [N, D].")
        if standard_probabilities.shape != candidate_probabilities.shape:
            raise ValueError("Probability tensors must have identical shapes.")
        if image_features.shape[0] != standard_probabilities.shape[0]:
            raise ValueError("Features and probabilities must have the same batch size.")

        max_standard, pred_standard = standard_probabilities.max(dim=1)
        max_candidate, pred_candidate = candidate_probabilities.max(dim=1)
        accepted = (
            (pred_standard == pred_candidate)
            & (max_standard > self.confidence_threshold)
            & (max_candidate > self.confidence_threshold)
        )

        for index in accepted.nonzero(as_tuple=False).flatten():
            class_index = int(pred_standard[index].item())
            self.feature_sum[class_index] += image_features[index].float()
            self.sample_count[class_index] += 1.0

    def centroids(self) -> torch.Tensor:
        """Return L2-normalized centroids; unseen classes remain zero."""

        counts = self.sample_count.clamp(min=1).unsqueeze(1)
        return functional.normalize(self.feature_sum / counts, dim=-1)


def build_scoring_directions(
    precision_matrix: torch.Tensor,
    distribution_means: torch.Tensor,
    consensus: ConsensusAccumulator,
    min_consensus_count: int,
) -> torch.Tensor:
    """Build one normalized Mahalanobis scoring direction per class."""

    if min_consensus_count < 1:
        raise ValueError("min_consensus_count must be positive.")
    if precision_matrix.ndim != 2 or precision_matrix.shape[0] != precision_matrix.shape[1]:
        raise ValueError("precision_matrix must be square.")
    if distribution_means.ndim != 2:
        raise ValueError("distribution_means must have shape [C, D].")

    centroids = consensus.centroids()
    directions = []
    for class_index in range(distribution_means.shape[0]):
        if consensus.sample_count[class_index].item() >= min_consensus_count:
            mean = centroids[class_index]
        else:
            mean = functional.normalize(distribution_means[class_index].float(), dim=-1)
        directions.append(functional.normalize(precision_matrix.float() @ mean, dim=-1))
    return torch.stack(directions)


def refine_prompt_weights(
    precision_matrix: torch.Tensor,
    distribution_means: torch.Tensor,
    consensus: ConsensusAccumulator,
    standard_embeddings: list[torch.Tensor],
    candidate_embeddings: list[torch.Tensor],
    selection_fraction: float = 0.10,
    min_consensus_count: int = 5,
) -> torch.Tensor:
    """Select discriminative prompts and return normalized weights with shape [D, C]."""

    if not 0.0 < selection_fraction <= 1.0:
        raise ValueError("selection_fraction must be in (0, 1].")
    if len(standard_embeddings) != len(candidate_embeddings):
        raise ValueError("Each class needs standard and candidate embeddings.")
    if len(standard_embeddings) != distribution_means.shape[0]:
        raise ValueError("Embedding lists must match the number of classes.")

    directions = build_scoring_directions(
        precision_matrix,
        distribution_means,
        consensus,
        min_consensus_count,
    )
    class_weights = []
    num_classes = len(standard_embeddings)

    for class_index, candidates in enumerate(candidate_embeddings):
        if candidates.ndim != 2 or candidates.shape[0] == 0:
            raise ValueError("Every candidate embedding tensor must have shape [M, D], M > 0.")
        positive = candidates @ directions[class_index]
        negative = (
            (candidates @ directions.T).sum(dim=1) - positive
        ) / max(num_classes - 1, 1)
        scores = positive - negative
        top_k = max(1, int(candidates.shape[0] * selection_fraction))
        top_indices = scores.topk(top_k).indices

        if scores[top_indices[0]].item() < 0:
            best_standard = (standard_embeddings[class_index] @ directions[class_index]).argmax()
            weight = standard_embeddings[class_index][best_standard]
        else:
            selected = candidates[top_indices]
            anchored = torch.cat([standard_embeddings[class_index], selected], dim=0)
            weight = functional.normalize(anchored.mean(dim=0), dim=-1)
        class_weights.append(weight)

    return torch.stack(class_weights, dim=1)
