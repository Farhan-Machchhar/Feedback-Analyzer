"""
Semantic Topic Clustering Service using Vector Embeddings, K-Means, and PCA.

Groups customer feedback into distinct operational problem clusters
and projects them into 2D coordinates for visual cluster exploration.
"""
from typing import List, Dict, Any
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from services.cache import semantic_cache
from core.config import logger


class TopicClusterer:
    """Clustering engine for unsupervised customer review grouping."""

    @staticmethod
    def cluster_reviews(reviews_data: List[Dict[str, Any]], max_clusters: int = 4) -> Dict[str, Any]:
        """
        Cluster a list of review dictionaries.
        Each item in reviews_data should have: 'review' (or 'text'), 'overall_label', 'overall_score', 'theme'.
        """
        if not reviews_data:
            return {"clusters": [], "points": []}

        n = len(reviews_data)
        # Determine appropriate k
        k = min(max_clusters, max(2, n // 3)) if n >= 4 else 1

        # Extract text and compute embeddings
        texts = [r.get("text") or r.get("review") or "" for r in reviews_data]
        vectors = np.array([semantic_cache._vectorize(t) for t in texts], dtype=np.float32)

        # 2D projection via PCA
        if n >= 2:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(vectors)
        else:
            coords = np.array([[0.0, 0.0]])

        # K-Means clustering
        if k > 1 and n >= k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            cluster_labels = kmeans.fit_predict(vectors)
        else:
            cluster_labels = np.zeros(n, dtype=int)

        # Label clusters meaningfully based on dominant themes and keywords
        cluster_names = {}
        for c_id in range(k):
            indices = [i for i, lbl in enumerate(cluster_labels) if lbl == c_id]
            themes = [reviews_data[i].get("theme", "general") for i in indices]
            most_common_theme = Counter(themes).most_common(1)[0][0] if themes else "general"
            cluster_names[c_id] = f"Cluster {c_id + 1}: {most_common_theme.replace('_', ' ').title()}"

        # Build output points
        points = []
        for i, r in enumerate(reviews_data):
            c_id = int(cluster_labels[i])
            points.append({
                "x": round(float(coords[i][0]), 3),
                "y": round(float(coords[i][1]), 3),
                "cluster_id": c_id,
                "cluster_name": cluster_names.get(c_id, f"Cluster {c_id+1}"),
                "review": texts[i][:100] + ("..." if len(texts[i]) > 100 else ""),
                "full_text": texts[i],
                "sentiment": r.get("overall_label") or r.get("label") or "neutral",
                "score": r.get("overall_score") or r.get("score") or 3,
                "theme": r.get("theme", "general"),
                "urgency": r.get("urgency", "low"),
            })

        # Summary statistics per cluster
        cluster_summaries = []
        for c_id in range(k):
            members = [p for p in points if p["cluster_id"] == c_id]
            avg_score = round(sum(m["score"] for m in members) / len(members), 1) if members else 0.0
            neg_count = sum(1 for m in members if m["sentiment"] == "negative")
            cluster_summaries.append({
                "cluster_id": c_id,
                "name": cluster_names.get(c_id, f"Cluster {c_id+1}"),
                "size": len(members),
                "average_score": avg_score,
                "negative_reviews": neg_count,
            })

        return {
            "num_clusters": k,
            "clusters": cluster_summaries,
            "points": points,
        }


clusterer = TopicClusterer()

