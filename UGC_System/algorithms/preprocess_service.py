from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import torch

from .utils import load_module


@dataclass
class PreprocessResult:
    dataset: str
    out_subdir: str
    output_dir: str
    edge_index_path: str
    node_feat_path: str
    label_path: str
    num_nodes: int
    num_edges: int
    feature_size: int
    num_classes: int
    x_is_dense: bool


class PreprocessService:
    def __init__(self, scripts_dir: Path) -> None:
        self.scripts_dir = scripts_dir

    def run(
        self,
        strategy: str,
        data_root: Path,
        dataset: str,
        out_subdir: str,
        jaccard_threshold: float = 0.5,
        force_undirected: bool | None = None,
        densify_limit: int = 50_000_000,
        max_dense_gb: float = 4.0,
        edge_dtype: str = "int32",
    ) -> PreprocessResult:
        if strategy == "v1":
            return self._run_common(
                module_name="ugc_data_process1",
                module_path=self.scripts_dir / "data_process" / "data_process1.py",
                data_root=data_root,
                dataset=dataset,
                out_subdir=out_subdir,
                force_undirected=force_undirected,
                densify_limit=densify_limit,
                max_dense_gb=max_dense_gb,
                edge_dtype=edge_dtype,
            )
        if strategy == "v2":
            return self._run_common(
                module_name="ugc_data_process2",
                module_path=self.scripts_dir / "data_process" / "data_process2.py",
                data_root=data_root,
                dataset=dataset,
                out_subdir=out_subdir,
                force_undirected=force_undirected,
                densify_limit=densify_limit,
                max_dense_gb=max_dense_gb,
                edge_dtype=edge_dtype,
            )
        if strategy == "v3":
            return self._run_common(
                module_name="ugc_data_process3",
                module_path=self.scripts_dir / "data_process" / "data_process3.py",
                data_root=data_root,
                dataset=dataset,
                out_subdir=out_subdir,
                force_undirected=force_undirected,
                densify_limit=densify_limit,
                max_dense_gb=max_dense_gb,
                edge_dtype=edge_dtype,
            )
        if strategy == "v4":
            return self.run_v4(
                data_root=data_root,
                dataset=dataset,
                out_subdir=out_subdir,
                jaccard_threshold=jaccard_threshold,
                force_undirected=force_undirected,
                densify_limit=densify_limit,
                max_dense_gb=max_dense_gb,
                edge_dtype=edge_dtype,
            )
        raise ValueError(f"Unsupported preprocess strategy: {strategy}")

    def load_builtin_processed_dataset(self, data_root: Path, dataset: str, out_subdir: str = "builtin_pt") -> PreprocessResult:
        processed_path = data_root / dataset / "processed" / "data.pt"
        if not processed_path.exists():
            raise FileNotFoundError(f"Missing processed dataset file: {processed_path}")

        data_dict, _ = self._torch_load(processed_path, weights_only=False)
        edge_index = data_dict["edge_index"]
        node_feat = data_dict["x"]
        label = data_dict["y"]

        out_dir = data_root / dataset / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(edge_index, out_dir / "edge_index.pt")
        torch.save(node_feat, out_dir / "node_feat.pt")
        torch.save(label.numpy(), out_dir / "label.pt")

        return PreprocessResult(
            dataset=dataset,
            out_subdir=out_subdir,
            output_dir=str(out_dir),
            edge_index_path=str(out_dir / "edge_index.pt"),
            node_feat_path=str(out_dir / "node_feat.pt"),
            label_path=str(out_dir / "label.pt"),
            num_nodes=int(node_feat.size(0)),
            num_edges=int(edge_index.size(1)),
            feature_size=int(node_feat.size(1)),
            num_classes=int(len(set(label.tolist()))),
            x_is_dense=bool(node_feat.layout == torch.strided),
        )

    def _run_common(
        self,
        module_name: str,
        module_path: Path,
        data_root: Path,
        dataset: str,
        out_subdir: str,
        force_undirected: bool | None,
        densify_limit: int,
        max_dense_gb: float,
        edge_dtype: str,
    ) -> PreprocessResult:
        module = load_module(module_name, module_path)
        module.process_one(
            dataset=dataset,
            root=data_root,
            out_subdir=out_subdir,
            force_undirected=force_undirected,
            densify_limit=densify_limit,
            max_dense_gb=max_dense_gb,
            edge_dtype=edge_dtype,
        )
        return self._build_result(dataset, out_subdir, data_root)

    def run_v4(
        self,
        data_root: Path,
        dataset: str,
        out_subdir: str = "ugc_v4",
        jaccard_threshold: float = 0.5,
        force_undirected: bool | None = None,
        densify_limit: int = 50_000_000,
        max_dense_gb: float = 4.0,
        edge_dtype: str = "int32",
    ) -> PreprocessResult:
        module = load_module(
            "ugc_data_process4",
            self.scripts_dir / "data_process" / "data_process4.py",
        )
        module.process_one(
            dataset=dataset,
            root=data_root,
            out_subdir=out_subdir,
            force_undirected=force_undirected,
            densify_limit=densify_limit,
            max_dense_gb=max_dense_gb,
            edge_dtype=edge_dtype,
            jaccard_threshold=jaccard_threshold,
        )
        return self._build_result(dataset, out_subdir, data_root)

    def _build_result(self, dataset: str, out_subdir: str, data_root: Path) -> PreprocessResult:
        out_dir = data_root / dataset / out_subdir
        edge_index = self._torch_load(out_dir / "edge_index.pt")
        node_feat = self._torch_load(out_dir / "node_feat.pt")
        label = self._torch_load(out_dir / "label.pt")

        return PreprocessResult(
            dataset=dataset,
            out_subdir=out_subdir,
            output_dir=str(out_dir),
            edge_index_path=str(out_dir / "edge_index.pt"),
            node_feat_path=str(out_dir / "node_feat.pt"),
            label_path=str(out_dir / "label.pt"),
            num_nodes=int(node_feat.size(0)),
            num_edges=int(edge_index.size(1)),
            feature_size=int(node_feat.size(1)),
            num_classes=int(len(set(label.tolist()))),
            x_is_dense=bool(node_feat.layout == torch.strided),
        )

    def _torch_load(self, path: Path, *, weights_only: bool = True):
        try:
            return torch.load(path, weights_only=weights_only)
        except pickle.UnpicklingError:
            if not weights_only:
                raise
            return torch.load(path, weights_only=False)
