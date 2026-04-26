from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class AnalysisResult:
    dataset: str
    data_root: str
    num_nodes: int
    num_edges: int
    avg_degree: float
    max_degree: int
    feature_dim: int
    label_type: str
    label_dim: int
    label_cardinality_max: int
    source: str


class AnalysisService:
    def analyze(self, dataset: str, data_root: Path) -> AnalysisResult:
        dataset_dir = data_root / dataset
        processed_path = dataset_dir / "processed" / "data.pt"
        raw_dir = dataset_dir / "raw"

        if processed_path.exists():
            return self._analyze_processed(dataset, data_root, processed_path)
        if raw_dir.exists():
            return self._analyze_raw(dataset, data_root, raw_dir)
        raise FileNotFoundError(f"Could not find analyzable data for dataset: {dataset_dir}")

    def _analyze_processed(self, dataset: str, data_root: Path, processed_path: Path) -> AnalysisResult:
        data_dict, _ = torch.load(processed_path)
        edge_index = data_dict["edge_index"]
        node_feat = data_dict["x"]
        labels = data_dict["y"]

        num_nodes = int(node_feat.size(0))
        num_edges = int(edge_index.size(1))
        avg_degree, max_degree = self._compute_degree_stats(edge_index, num_nodes)
        label_values = labels.tolist()
        label_dim = int(len(set(label_values)))

        return AnalysisResult(
            dataset=dataset,
            data_root=str(data_root),
            num_nodes=num_nodes,
            num_edges=num_edges,
            avg_degree=avg_degree,
            max_degree=max_degree,
            feature_dim=int(node_feat.size(1)),
            label_type="single",
            label_dim=label_dim,
            label_cardinality_max=1,
            source="processed",
        )

    def _analyze_raw(self, dataset: str, data_root: Path, raw_dir: Path) -> AnalysisResult:
        edge_path = raw_dir / "edgelist.txt"
        label_path = raw_dir / "labels.txt"
        feature_path = raw_dir / "attrs.npz"

        edge_index, unique_nodes = self._load_edge_index(edge_path)
        label_entries, labeled_nodes, unique_labels, label_cardinality_max = self._load_labels(label_path)
        feature_rows, feature_dim = self._load_feature_shape(feature_path)

        num_nodes = max(len(unique_nodes), len(labeled_nodes), feature_rows)
        num_edges = int(edge_index.shape[1])
        avg_degree, max_degree = self._compute_degree_stats(edge_index, num_nodes)

        return AnalysisResult(
            dataset=dataset,
            data_root=str(data_root),
            num_nodes=int(num_nodes),
            num_edges=num_edges,
            avg_degree=avg_degree,
            max_degree=max_degree,
            feature_dim=int(feature_dim),
            label_type="multi" if label_cardinality_max > 1 else "single",
            label_dim=int(len(unique_labels)),
            label_cardinality_max=int(label_cardinality_max),
            source="raw",
        )

    def _load_edge_index(self, edge_path: Path) -> tuple[np.ndarray, set[int]]:
        if not edge_path.exists():
            raise FileNotFoundError(f"Missing edge file: {edge_path}")
        edges: list[tuple[int, int]] = []
        unique_nodes: set[int] = set()
        with edge_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                src = int(parts[0])
                dst = int(parts[1])
                edges.append((src, dst))
                unique_nodes.add(src)
                unique_nodes.add(dst)
        if not edges:
            return np.empty((2, 0), dtype=np.int64), unique_nodes
        edge_index = np.asarray(edges, dtype=np.int64).T
        return edge_index, unique_nodes

    def _load_labels(self, label_path: Path) -> tuple[list[list[int]], set[int], set[int], int]:
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        entries: list[list[int]] = []
        labeled_nodes: set[int] = set()
        unique_labels: set[int] = set()
        label_cardinality_max = 0
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                node_id = int(parts[0])
                labels = [int(part) for part in parts[1:]]
                labeled_nodes.add(node_id)
                unique_labels.update(labels)
                label_cardinality_max = max(label_cardinality_max, len(labels))
                entries.append(labels)
        return entries, labeled_nodes, unique_labels, label_cardinality_max

    def _load_feature_shape(self, feature_path: Path) -> tuple[int, int]:
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
        try:
            from scipy import sparse

            matrix = sparse.load_npz(feature_path)
            return int(matrix.shape[0]), int(matrix.shape[1])
        except Exception:
            payload = np.load(feature_path, allow_pickle=True)
            if hasattr(payload, "files"):
                if "shape" in payload.files:
                    shape = tuple(int(value) for value in payload["shape"].tolist())
                    if len(shape) >= 2:
                        return int(shape[0]), int(shape[1])
                for key in payload.files:
                    value = payload[key]
                    if getattr(value, "ndim", 0) >= 2:
                        return int(value.shape[0]), int(value.shape[1])
                if payload.files:
                    value = payload[payload.files[0]]
                    shape = getattr(value, "shape", ())
                    if len(shape) >= 2:
                        return int(shape[0]), int(shape[1])
            raise RuntimeError(f"Unable to infer feature shape from {feature_path}")

    def _compute_degree_stats(self, edge_index: np.ndarray | torch.Tensor, num_nodes: int) -> tuple[float, int]:
        if isinstance(edge_index, torch.Tensor):
            edge_array = edge_index.detach().cpu().numpy()
        else:
            edge_array = edge_index
        if num_nodes <= 0:
            return 0.0, 0
        degrees = np.zeros(num_nodes, dtype=np.int64)
        if edge_array.size:
            src = edge_array[0]
            dst = edge_array[1]
            np.add.at(degrees, src, 1)
            np.add.at(degrees, dst, 1)
        avg_degree = float(degrees.mean())
        max_degree = int(degrees.max()) if degrees.size else 0
        return avg_degree, max_degree
