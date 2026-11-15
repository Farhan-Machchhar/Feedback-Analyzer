"""
Semantic and Exact Caching Service.

Uses vector cosine similarity over text representations to prevent
redundant LLM API calls on semantically equivalent customer reviews.
"""
from typing import Optional, Tuple, List, Dict, Any
import re
import numpy as np
from models.schemas import AnalysisResult
from core.config import settings, logger


class SemanticCache:
    """
    In-memory semantic cache with vector similarity lookup.
    Tracks hit/miss metrics and estimated dollar savings.
    """

    def __init__(
        self,
        similarity_threshold: float = settings.CACHE_SIMILARITY_THRESHOLD,
        max_entries: int = settings.CACHE_MAX_ENTRIES,
    ):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.texts: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.results: List[AnalysisResult] = []
        self.exact_map: Dict[str, AnalysisResult] = {}
        
        # Metrics
        self.hits: int = 0
        self.misses: int = 0
        self.cost_per_review_usd: float = 0.0004  # Approx Gemini Flash input+output cost

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _vectorize(self, text: str, dim: int = 256) -> np.ndarray:
        """
        Fast deterministic sub-word hashing embedding (dimension 256).
        Zero external dependencies, fast (<0.1ms), normalized to unit length.
        Captures morphological stems and n-gram semantics accurately.
        """
        clean = self._normalize_text(text)
        vec = np.zeros(dim, dtype=np.float32)
        
        # Word and character 3-grams
        tokens = clean.split()
        for tok in tokens:
            idx = hash("w_" + tok) % dim
            vec[idx] += 2.0
            
        for i in range(len(clean) - 2):
            trigram = clean[i:i+3]
            idx = hash("c_" + trigram) % dim
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def get(self, text: str) -> Optional[Tuple[AnalysisResult, float]]:
        """
        Check for an exact or semantic match.
        Returns: (cached_result, similarity_score) or None.
        """
        clean = self._normalize_text(text)
        
        # 1. Exact match (O(1))
        if clean in self.exact_map:
            self.hits += 1
            result = self.exact_map[clean].model_copy(deep=True)
            result.cached = True
            result.source = "cache_exact"
            return result, 1.0

        # 2. Semantic vector match
        if not self.vectors:
            self.misses += 1
            return None

        query_vec = self._vectorize(clean)
        sims = [float(np.dot(query_vec, v)) for v in self.vectors]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= self.threshold:
            self.hits += 1
            cached_item = self.results[best_idx].model_copy(deep=True)
            cached_item.cached = True
            cached_item.source = "cache_semantic"
            logger.info(
                f"Semantic Cache HIT (similarity: {best_sim:.3f}): '{text[:30]}...' -> '{self.texts[best_idx][:30]}...'"
            )
            return cached_item, best_sim

        self.misses += 1
        return None

    def put(self, text: str, result: AnalysisResult):
        """Store a analyzed review in the cache."""
        clean = self._normalize_text(text)
        self.exact_map[clean] = result

        if len(self.texts) >= self.max_entries:
            # Evict oldest
            self.texts.pop(0)
            self.vectors.pop(0)
            self.results.pop(0)

        vec = self._vectorize(clean)
        self.texts.append(clean)
        self.vectors.append(vec)
        self.results.append(result)

    @property
    def metrics(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": round(hit_rate, 2),
            "estimated_cost_saved_usd": round(self.hits * self.cost_per_review_usd, 4),
            "cached_entries": len(self.texts),
        }

    def clear(self):
        self.texts.clear()
        self.vectors.clear()
        self.results.clear()
        self.exact_map.clear()
        self.hits = 0
        self.misses = 0


# Global singleton instance
semantic_cache = SemanticCache()

