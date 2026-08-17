"""Validated constants for the paper protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass

PAPER_DATASETS = (
    "fgvc",
    "caltech101",
    "stanford_cars",
    "dtd",
    "eurosat",
    "oxford_flowers",
    "food101",
    "oxford_pets",
    "sun397",
    "ucf101",
)


@dataclass(frozen=True)
class CGPSConfig:
    """CGPS hyperparameters used for every main-table dataset."""

    seed: int = 42
    backbone: str = "ViT-B/16"
    batch_size: int = 1
    confidence_threshold: float = 0.30
    trigger_fraction: float = 0.25
    selection_fraction: float = 0.10
    min_consensus_count: int = 5

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError("The paper protocol requires batch_size=1.")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("trigger_fraction", self.trigger_fraction),
            ("selection_fraction", self.selection_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        if self.min_consensus_count < 1:
            raise ValueError("min_consensus_count must be positive.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PAPER_CONFIG = CGPSConfig()


def parse_datasets(value: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated dataset list."""

    datasets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not datasets:
        raise ValueError("At least one dataset is required.")
    unknown = sorted(set(datasets) - set(PAPER_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported paper dataset(s): {', '.join(unknown)}")
    return datasets
