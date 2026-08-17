import pytest

from cgps.config import PAPER_CONFIG, PAPER_DATASETS, CGPSConfig, parse_datasets


def test_paper_defaults_are_frozen() -> None:
    assert PAPER_CONFIG == CGPSConfig(
        seed=42,
        backbone="ViT-B/16",
        batch_size=1,
        confidence_threshold=0.30,
        trigger_fraction=0.25,
        selection_fraction=0.10,
        min_consensus_count=5,
    )
    assert len(PAPER_DATASETS) == 10


def test_dataset_parser_rejects_non_paper_dataset() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_datasets("dtd,unknown")
