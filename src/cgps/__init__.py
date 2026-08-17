"""Reusable CGPS components."""

from .config import PAPER_CONFIG, CGPSConfig
from .core import ConsensusAccumulator, refine_prompt_weights

__all__ = ["CGPSConfig", "ConsensusAccumulator", "PAPER_CONFIG", "refine_prompt_weights"]
__version__ = "1.0.0"
