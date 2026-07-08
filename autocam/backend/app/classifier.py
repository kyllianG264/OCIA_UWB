"""
classifier.py
Gère le clustering des features biomécaniques.
Inclut normalisation, sélection automatique du nombre de groupes,
détection des clips ambigus et sauvegarde/chargement du modèle.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class ClusterManager:
    """
    Encapsule KMeans + StandardScaler pour le clustering des athlètes.
    Le scaler est toujours appliqué avant le clustering.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        algorithm: str = "kmeans_auto",
        n_init: int = 20,
        max_iter: int = 500,
        random_state: int = 42,
        max_auto_clusters: int = 8,
        min_silhouette_score: float = 0.0,
        manual_distance_factor: float = 1.5,
        min_clusters: int = 0,
    ):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_auto_clusters = max_auto_clusters
        self.min_silhouette_score = min_silhouette_score
        self.manual_distance_factor = manual_distance_factor
        self.min_clusters = min_clusters
        self.scaler = StandardScaler()
        self.cluster_centers_: np.ndarray | None = None
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Entraîne le scaler + K-Means sur la matrice de features.

        Args:
            X: (n_clips, n_features)

        Returns:
            labels: (n_clips,) — entier de cluster pour chaque clip.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.n_clusters = int(self.model.n_clusters)
        return self.model.labels_

    def fit_auto(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Détermine automatiquement un nombre de groupes plausible, puis détecte
        les clips ambigus qui doivent être revus manuellement.

        Returns:
            labels: (n_clips,) — cluster attribué à chaque clip
            distances: (n_clips,) — distance au centroïde attribué
            manual_mask: (n_clips,) — True si le clip doit être trié manuellement
            metadata: informations de diagnostic sur le clustering
        """
        sample_count = len(X)
        if sample_count == 0:
            raise ValueError("Cannot cluster an empty feature matrix.")

        X_scaled = self.scaler.fit_transform(X)
        labels, assigned_distances, algo_meta = self._fit_with_algorithm(X_scaled)

        non_noise_mask = labels >= 0
        valid_labels = labels[non_noise_mask]
        valid_X = X_scaled[non_noise_mask]

        silhouette_values = np.full(sample_count, -1.0, dtype=float)
        unique_valid_clusters = np.unique(valid_labels) if valid_labels.size else np.array([])

        if unique_valid_clusters.size > 1 and valid_X.shape[0] > unique_valid_clusters.size:
            sil_valid = silhouette_samples(valid_X, valid_labels)
            silhouette_values[non_noise_mask] = sil_valid
            global_silhouette = float(silhouette_score(valid_X, valid_labels))
        else:
            silhouette_values[non_noise_mask] = 1.0 if valid_X.shape[0] > 1 else 0.0
            global_silhouette = 1.0 if valid_X.shape[0] > 1 else 0.0

        manual_mask = labels < 0
        if np.any(non_noise_mask):
            manual_mask_non_noise = self._build_manual_mask(
                labels[non_noise_mask],
                assigned_distances[non_noise_mask],
                silhouette_values[non_noise_mask],
            )
            manual_mask[non_noise_mask] |= manual_mask_non_noise

        metadata = {
            "algorithm": self.algorithm,
            "selected_clusters": int(self.n_clusters),
            "global_silhouette": global_silhouette,
            "manual_clip_count": int(np.sum(manual_mask)),
            "estimated_eps": self._estimate_neighbor_radius(X_scaled),
            "min_clusters_applied": int(self.min_clusters if self.min_clusters > 0 else max(2, int(round(np.sqrt(sample_count / 2.0))))),
        }
        metadata.update(algo_meta)
        return labels, assigned_distances, manual_mask, metadata

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit le cluster pour de nouveaux clips.

        Args:
            X: (n_clips, n_features)

        Returns:
            labels: (n_clips,)
        """
        if not self.is_fitted:
            raise RuntimeError("ClusterManager must be fitted before predicting.")
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict"):
            return self.model.predict(X_scaled)
        if self.cluster_centers_ is None or self.cluster_centers_.size == 0:
            raise RuntimeError("Loaded clustering model does not support prediction.")

        delta = X_scaled[:, None, :] - self.cluster_centers_[None, :, :]
        dists = np.linalg.norm(delta, axis=2)
        return np.argmin(dists, axis=1)

    def cluster_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne la distance de chaque clip à son centroïde (proxy de confiance).
        Distance faible = clip très représentatif du groupe.

        Args:
            X: (n_clips, n_features)

        Returns:
            distances: (n_clips,) — distance au centroïde attribué
        """
        if not self.is_fitted:
            raise RuntimeError("ClusterManager must be fitted before computing distances.")
        X_scaled = self.scaler.transform(X)
        labels = self.predict(X)

        if hasattr(self.model, "transform"):
            all_distances = self.model.transform(X_scaled)  # (n_clips, n_clusters)
            return all_distances[np.arange(len(labels)), labels]

        if self.cluster_centers_ is None or self.cluster_centers_.size == 0:
            raise RuntimeError("Loaded clustering model does not provide cluster distances.")

        delta = X_scaled[:, None, :] - self.cluster_centers_[None, :, :]
        dists = np.linalg.norm(delta, axis=2)
        return dists[np.arange(len(labels)), labels]

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modèle complet (scaler + kmeans) en pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "model": self.model,
                    "n_clusters": self.n_clusters,
                    "algorithm": self.algorithm,
                    "is_fitted": self.is_fitted,
                    "n_init": self.n_init,
                    "max_iter": self.max_iter,
                    "random_state": self.random_state,
                    "max_auto_clusters": self.max_auto_clusters,
                    "min_silhouette_score": self.min_silhouette_score,
                    "manual_distance_factor": self.manual_distance_factor,
                    "min_clusters": self.min_clusters,
                    "cluster_centers": self.cluster_centers_,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ClusterManager":
        """Charge un modèle depuis un fichier pickle."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        manager = cls(
            n_clusters=data["n_clusters"],
            algorithm=data.get("algorithm", "kmeans_auto"),
            n_init=data.get("n_init", 20),
            max_iter=data.get("max_iter", 500),
            random_state=data.get("random_state", 42),
            max_auto_clusters=data.get("max_auto_clusters", 8),
            min_silhouette_score=data.get("min_silhouette_score", 0.0),
            manual_distance_factor=data.get("manual_distance_factor", 1.5),
            min_clusters=data.get("min_clusters", 0),
        )
        manager.scaler = data["scaler"]
        manager.model = data["model"]
        manager.is_fitted = data["is_fitted"]
        manager.n_clusters = data["n_clusters"]
        manager.cluster_centers_ = data.get("cluster_centers")
        return manager

    def _resolve_effective_cluster_count(self, X_scaled: np.ndarray) -> int:
        sample_count = len(X_scaled)
        if sample_count <= 2:
            return 1 if sample_count == 1 else 2

        if self.min_clusters > 0:
            return max(2, min(int(self.min_clusters), sample_count - 1))

        return self._select_cluster_count(X_scaled)

    def _compute_cluster_centers(self, X_scaled: np.ndarray, labels: np.ndarray) -> np.ndarray:
        centers: list[np.ndarray] = []
        for cluster_id in sorted(int(v) for v in np.unique(labels) if v >= 0):
            mask = labels == cluster_id
            if np.any(mask):
                centers.append(np.mean(X_scaled[mask], axis=0))
        if not centers:
            return np.zeros((0, X_scaled.shape[1]), dtype=float)
        return np.vstack(centers)

    def _assigned_distances_from_centers(self, X_scaled: np.ndarray, labels: np.ndarray) -> np.ndarray:
        distances = np.full(len(labels), np.inf, dtype=float)
        for cluster_id in sorted(int(v) for v in np.unique(labels) if v >= 0):
            mask = labels == cluster_id
            if not np.any(mask):
                continue
            center = np.mean(X_scaled[mask], axis=0)
            distances[mask] = np.linalg.norm(X_scaled[mask] - center, axis=1)
        return distances

    def _fit_with_algorithm(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        sample_count = len(X_scaled)
        algo = str(self.algorithm or "kmeans_auto").lower()
        meta: dict = {}

        if sample_count <= 1:
            self.model = KMeans(n_clusters=1, random_state=self.random_state, n_init=1, max_iter=max(10, self.max_iter))
            self.model.fit(X_scaled)
            labels = np.zeros(sample_count, dtype=int)
            assigned_distances = np.zeros(sample_count, dtype=float)
            self.cluster_centers_ = self.model.cluster_centers_
            self.n_clusters = 1
            self.is_fitted = True
            meta["selected_clusters"] = 1
            meta["tiny_sample_fallback"] = True
            return labels, assigned_distances, meta

        if algo in {"kmeans_auto", "kmeans", "kmeans_fixed"}:
            best_k = self._resolve_effective_cluster_count(X_scaled) if algo == "kmeans_auto" else max(2, min(self.n_clusters, sample_count))
            self.model = KMeans(
                n_clusters=best_k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            self.model.fit(X_scaled)
            labels = self.model.labels_.astype(int)
            all_distances = self.model.transform(X_scaled)
            assigned_distances = all_distances[np.arange(sample_count), labels]
            self.cluster_centers_ = self.model.cluster_centers_
            self.n_clusters = int(self.model.n_clusters)
            meta["selected_clusters"] = int(self.n_clusters)

        elif algo == "spectral":
            best_k = self._resolve_effective_cluster_count(X_scaled)
            self.model = SpectralClustering(
                n_clusters=best_k,
                random_state=self.random_state,
                n_init=max(1, min(self.n_init, 10)),
                assign_labels="kmeans",
            )
            labels = self.model.fit_predict(X_scaled).astype(int)
            self.cluster_centers_ = self._compute_cluster_centers(X_scaled, labels)
            assigned_distances = self._assigned_distances_from_centers(X_scaled, labels)
            self.n_clusters = int(len(np.unique(labels[labels >= 0])))
            meta["selected_clusters"] = int(self.n_clusters)

        elif algo in {"agglomerative", "hierarchical"}:
            best_k = self._resolve_effective_cluster_count(X_scaled)
            self.model = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
            labels = self.model.fit_predict(X_scaled).astype(int)
            self.cluster_centers_ = self._compute_cluster_centers(X_scaled, labels)
            assigned_distances = self._assigned_distances_from_centers(X_scaled, labels)
            self.n_clusters = int(len(np.unique(labels[labels >= 0])))
            meta["selected_clusters"] = int(self.n_clusters)

        elif algo == "gmm":
            best_k = self._resolve_effective_cluster_count(X_scaled)
            self.model = GaussianMixture(
                n_components=best_k,
                random_state=self.random_state,
                n_init=max(1, min(self.n_init, 10)),
                max_iter=self.max_iter,
            )
            self.model.fit(X_scaled)
            labels = self.model.predict(X_scaled).astype(int)
            self.cluster_centers_ = np.asarray(self.model.means_, dtype=float)
            assigned_distances = self._assigned_distances_from_centers(X_scaled, labels)
            self.n_clusters = int(best_k)
            meta["selected_clusters"] = int(self.n_clusters)

        elif algo == "dbscan":
            eps = self._estimate_neighbor_radius(X_scaled)
            if eps <= 0:
                eps = 0.5
            self.model = DBSCAN(eps=float(eps), min_samples=3)
            labels = self.model.fit_predict(X_scaled).astype(int)
            self.cluster_centers_ = self._compute_cluster_centers(X_scaled, labels)
            assigned_distances = self._assigned_distances_from_centers(X_scaled, labels)
            self.n_clusters = int(len(np.unique(labels[labels >= 0])))
            meta["selected_clusters"] = int(self.n_clusters)
            meta["dbscan_eps"] = float(eps)
            meta["dbscan_noise_count"] = int(np.sum(labels < 0))

        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")

        self.is_fitted = True
        return labels, assigned_distances, meta

    def _gap_statistic(
        self, X_scaled: np.ndarray, max_k: int, n_refs: int = 10
    ) -> tuple[list[float], list[float]]:
        """
        Calcule le gap statistic (Tibshirani et al.) pour k=2..max_k.
        Retourne (gaps, s_values) où gaps[i] correspond à k = i+2.
        Moins biaisé que le silhouette score envers k=2.
        """
        rng = np.random.default_rng(self.random_state)
        n, d = X_scaled.shape
        col_mins = X_scaled.min(axis=0)
        col_maxs = X_scaled.max(axis=0)

        gaps: list[float] = []
        s_values: list[float] = []

        for k in range(2, max_k + 1):
            model_k = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            model_k.fit(X_scaled)
            log_wk = np.log(max(model_k.inertia_, 1e-10))

            ref_log_wks: list[float] = []
            for _ in range(n_refs):
                ref = rng.uniform(col_mins, col_maxs, size=(n, d))
                ref_model = KMeans(
                    n_clusters=k,
                    random_state=int(rng.integers(0, 9999)),
                    n_init=5,
                    max_iter=100,
                )
                ref_model.fit(ref)
                ref_log_wks.append(np.log(max(ref_model.inertia_, 1e-10)))

            expected = float(np.mean(ref_log_wks))
            sdk = float(np.std(ref_log_wks))
            gaps.append(expected - log_wk)
            s_values.append(sdk * np.sqrt(1 + 1.0 / n_refs))

        return gaps, s_values

    def _select_cluster_count(self, X_scaled: np.ndarray) -> int:
        sample_count = len(X_scaled)
        if sample_count <= 2:
            return 1 if sample_count == 1 else 2

        max_k = min(self.max_auto_clusters, sample_count - 1)
        if max_k < 2:
            return 1

        # Borne inférieure: heuristique sqrt(n/2), ou valeur explicite si définie
        auto_min_k = max(2, int(round(np.sqrt(sample_count / 2.0))))
        min_k = self.min_clusters if self.min_clusters > 0 else auto_min_k
        min_k = max(2, min(min_k, max_k))

        gaps, s_values = self._gap_statistic(X_scaled, max_k=max_k)
        # gaps[i] → k = i + 2

        # Critère de Tibshirani: premier k >= min_k tel que gap(k) >= gap(k+1) - s(k+1)
        best_k = min_k
        for k in range(min_k, max_k):
            idx = k - 2
            next_idx = idx + 1
            if next_idx < len(gaps) and gaps[idx] >= gaps[next_idx] - s_values[next_idx]:
                best_k = k
                break
        else:
            # Fallback: k avec le plus grand gap dans [min_k, max_k]
            candidates = [gaps[k - 2] for k in range(min_k, max_k + 1) if k - 2 < len(gaps)]
            if candidates:
                best_k = min_k + int(np.argmax(candidates))

        return best_k

    def _build_manual_mask(
        self,
        labels: np.ndarray,
        assigned_distances: np.ndarray,
        silhouette_values: np.ndarray,
    ) -> np.ndarray:
        manual_mask = silhouette_values < self.min_silhouette_score

        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_distances = assigned_distances[cluster_mask]
            if cluster_distances.size < 2:
                continue

            median_distance = float(np.median(cluster_distances))
            mad = float(np.median(np.abs(cluster_distances - median_distance)))
            robust_spread = max(mad, 1e-6)
            distance_threshold = median_distance + (self.manual_distance_factor * robust_spread)
            manual_mask[cluster_mask] |= cluster_distances > distance_threshold

        return manual_mask.astype(bool)

    def _estimate_neighbor_radius(self, X_scaled: np.ndarray) -> float:
        sample_count = len(X_scaled)
        if sample_count < 2:
            return 0.0

        neighbor_count = min(2, sample_count)
        neighbors = NearestNeighbors(n_neighbors=neighbor_count)
        neighbors.fit(X_scaled)
        distances, _ = neighbors.kneighbors(X_scaled)
        if distances.shape[1] < 2:
            return 0.0
        return float(np.median(distances[:, 1]))
