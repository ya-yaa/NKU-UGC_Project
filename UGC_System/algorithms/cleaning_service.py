from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CleaningResult:
    dataset: str
    source_data_root: str
    cleaned_data_root: str
    cleaned_dataset: str
    num_nodes_before: int
    num_nodes_after: int
    num_edges_before: int
    num_edges_after: int
    removed_unlabeled_nodes: int
    removed_duplicate_edges: int
    skipped: bool
    reason: str | None = None


class CleaningService:
    def clean(self, dataset: str, source_root: Path, cleaned_root: Path) -> CleaningResult:
        dataset_dir = source_root / dataset
        processed_path = dataset_dir / "processed" / "data.pt"
        raw_dir = dataset_dir / "raw"

        if processed_path.exists():
            return CleaningResult(
                dataset=dataset,
                source_data_root=str(source_root),
                cleaned_data_root=str(source_root),
                cleaned_dataset=dataset,
                num_nodes_before=0,
                num_nodes_after=0,
                num_edges_before=0,
                num_edges_after=0,
                removed_unlabeled_nodes=0,
                removed_duplicate_edges=0,
                skipped=True,
                reason="processed dataset; cleaning skipped",
            )

        if not raw_dir.exists():
            raise FileNotFoundError(f"Missing raw dataset directory: {raw_dir}")

        edge_path = raw_dir / "edgelist.txt"
        label_path = raw_dir / "labels.txt"
        feature_path = raw_dir / "attrs.npz"
        if not edge_path.exists() or not label_path.exists() or not feature_path.exists():
            raise FileNotFoundError(f"Cleaning requires edgelist.txt, labels.txt and attrs.npz under {raw_dir}")

        label_entries = self._load_label_entries(label_path)
        labeled_nodes = sorted(label_entries.keys())
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(labeled_nodes)}

        edges_before = 0
        dedup_edges: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()
        nodes_in_edges_before: set[int] = set()
        with edge_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                src = int(parts[0])
                dst = int(parts[1])
                edges_before += 1
                nodes_in_edges_before.add(src)
                nodes_in_edges_before.add(dst)
                if src not in node_mapping or dst not in node_mapping:
                    continue
                remapped = (node_mapping[src], node_mapping[dst])
                if remapped in seen_edges:
                    continue
                seen_edges.add(remapped)
                dedup_edges.append(remapped)

        feature_rows, save_sparse = self._slice_features(feature_path, labeled_nodes)

        cleaned_raw_dir = cleaned_root / dataset / "raw"
        cleaned_raw_dir.mkdir(parents=True, exist_ok=True)

        with (cleaned_raw_dir / "edgelist.txt").open("w", encoding="utf-8") as handle:
            for src, dst in dedup_edges:
                handle.write(f"{src} {dst}\n")

        with (cleaned_raw_dir / "labels.txt").open("w", encoding="utf-8") as handle:
            for old_id in labeled_nodes:
                new_id = node_mapping[old_id]
                labels = label_entries[old_id]
                handle.write(f"{new_id} {' '.join(str(label) for label in labels)}\n")

        if save_sparse:
            from scipy import sparse

            sparse.save_npz(cleaned_raw_dir / "attrs.npz", feature_rows)
        else:
            np.savez_compressed(cleaned_raw_dir / "attrs.npz", arr_0=feature_rows)

        num_nodes_before = max(max(nodes_in_edges_before, default=-1) + 1, len(labeled_nodes))
        num_nodes_after = len(labeled_nodes)
        num_edges_after = len(dedup_edges)
        removed_unlabeled_nodes = max(0, num_nodes_before - num_nodes_after)
        removed_duplicate_edges = max(0, edges_before - num_edges_after)

        return CleaningResult(
            dataset=dataset,
            source_data_root=str(source_root),
            cleaned_data_root=str(cleaned_root),
            cleaned_dataset=dataset,
            num_nodes_before=num_nodes_before,
            num_nodes_after=num_nodes_after,
            num_edges_before=edges_before,
            num_edges_after=num_edges_after,
            removed_unlabeled_nodes=removed_unlabeled_nodes,
            removed_duplicate_edges=removed_duplicate_edges,
            skipped=False,
        )

    def _load_label_entries(self, label_path: Path) -> dict[int, list[int]]:
        entries: dict[int, list[int]] = {}
        with label_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                node_id = int(parts[0])
                labels = [int(value) for value in parts[1:]]
                if not labels:
                    continue
                entries[node_id] = labels
        return entries

    def _slice_features(self, feature_path: Path, labeled_nodes: list[int]) -> tuple[object, bool]:
        try:
            from scipy import sparse

            matrix = sparse.load_npz(feature_path)
            return matrix[labeled_nodes], True
        except Exception:
            payload = np.load(feature_path, allow_pickle=False)
            if "arr_0" in payload.files:
                return payload["arr_0"][labeled_nodes].astype(np.float32, copy=False), False
            if {"data", "indices", "indptr", "shape"}.issubset(set(payload.files)):
                from scipy import sparse

                shape = tuple(int(value) for value in payload["shape"].tolist())
                matrix = sparse.csr_matrix(
                    (payload["data"], payload["indices"], payload["indptr"]),
                    shape=shape,
                )
                return matrix[labeled_nodes], True
            raise RuntimeError(f"Unsupported attrs format: {feature_path}")
