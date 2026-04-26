from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import load_module


@dataclass
class AlphaResult:
    dataset: str
    strategy: str
    alpha: float
    labeled_edges: int
    total_edges: int
    coverage_percent: float
    jaccard_threshold: float | None = None


class AlphaService:
    def __init__(self, scripts_dir: Path) -> None:
        self.scripts_dir = scripts_dir

    def calculate(self, strategy: str, data_root: Path, dataset: str, jaccard_threshold: float = 0.5) -> AlphaResult:
        if strategy == "v1":
            return self.calculate_v1(data_root, dataset)
        if strategy == "v2":
            return self.calculate_v2(data_root, dataset)
        if strategy == "v3":
            return self.calculate_v3(data_root, dataset)
        if strategy == "v4":
            return self.calculate_v4(data_root, dataset, jaccard_threshold)
        raise ValueError(f"Unsupported alpha strategy: {strategy}")

    def calculate_v1(self, data_root: Path, dataset: str) -> AlphaResult:
        module = load_module(
            "ugc_get_alpha1",
            self.scripts_dir / "get_alpha" / "get_alpha1.py",
        )
        raw_dir = data_root / dataset / "raw"
        label_dict = module.load_first_labels(raw_dir / "labels.txt")
        result = module.calculate_alpha(raw_dir / "edgelist.txt", label_dict)
        if result is None:
            raise RuntimeError(f"Failed to calculate alpha for {dataset}")
        alpha, labeled_edges, total_edges = result
        coverage = (labeled_edges / total_edges * 100.0) if total_edges else 0.0
        return AlphaResult(
            dataset=dataset,
            strategy="v1",
            alpha=float(alpha),
            labeled_edges=int(labeled_edges),
            total_edges=int(total_edges),
            coverage_percent=float(coverage),
        )

    def calculate_v2(self, data_root: Path, dataset: str) -> AlphaResult:
        module = load_module(
            "ugc_get_alpha2",
            self.scripts_dir / "get_alpha" / "get_alpha2.py",
        )
        raw_dir = data_root / dataset / "raw"
        label_sets = module.load_label_sets(raw_dir / "labels.txt")
        result = module.calculate_alpha_v2(raw_dir / "edgelist.txt", label_sets)
        if result is None:
            raise RuntimeError(f"Failed to calculate alpha for {dataset}")
        alpha, labeled_edges, total_edges = result
        coverage = (labeled_edges / total_edges * 100.0) if total_edges else 0.0
        return AlphaResult(
            dataset=dataset,
            strategy="v2",
            alpha=float(alpha),
            labeled_edges=int(labeled_edges),
            total_edges=int(total_edges),
            coverage_percent=float(coverage),
        )

    def calculate_v3(self, data_root: Path, dataset: str) -> AlphaResult:
        module = load_module(
            "ugc_get_alpha3",
            self.scripts_dir / "get_alpha" / "get_alpha3.py",
        )
        raw_dir = data_root / dataset / "raw"
        label_lists = module.load_label_lists(raw_dir / "labels.txt")
        result = module.calculate_alpha_v3(raw_dir / "edgelist.txt", label_lists)
        if result is None:
            raise RuntimeError(f"Failed to calculate alpha for {dataset}")
        alpha, labeled_edges, total_edges = result
        coverage = (labeled_edges / total_edges * 100.0) if total_edges else 0.0
        return AlphaResult(
            dataset=dataset,
            strategy="v3",
            alpha=float(alpha),
            labeled_edges=int(labeled_edges),
            total_edges=int(total_edges),
            coverage_percent=float(coverage),
        )

    def calculate_v4(self, data_root: Path, dataset: str, jaccard_threshold: float = 0.5) -> AlphaResult:
        module = load_module(
            "ugc_get_alpha4",
            self.scripts_dir / "get_alpha" / "get_alpha4.py",
        )
        raw_dir = data_root / dataset / "raw"
        label_sets = module.load_label_sets(raw_dir / "labels.txt")
        result = module.calculate_alpha_v4(raw_dir / "edgelist.txt", label_sets, jaccard_threshold)
        if result is None:
            raise RuntimeError(f"Failed to calculate alpha for {dataset}")
        alpha, labeled_edges, total_edges = result
        coverage = (labeled_edges / total_edges * 100.0) if total_edges else 0.0
        return AlphaResult(
            dataset=dataset,
            strategy="v4",
            alpha=float(alpha),
            labeled_edges=int(labeled_edges),
            total_edges=int(total_edges),
            coverage_percent=float(coverage),
            jaccard_threshold=float(jaccard_threshold),
        )
