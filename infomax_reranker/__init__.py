"""InfoMax reranker: information-maximising passage selection over a retrieved candidate pool."""
from .core import TUNED, by_score, fused, infomax, infomax_select, mmr, rank
from .neural import InfoMaxReranker

__all__ = ["InfoMaxReranker", "TUNED", "by_score", "fused", "infomax", "infomax_select", "mmr", "rank"]
__version__ = "0.1.0"
