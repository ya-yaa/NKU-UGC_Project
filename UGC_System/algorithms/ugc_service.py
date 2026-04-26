from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .utils import parse_float, parse_int, run_subprocess


@dataclass
class UGCResult:
    dataset: str
    model_type: str
    epochs: int
    bin_width: float
    coarsen_only: bool
    reduction_percent: float | None
    supernodes: int | None
    original_nodes: int | None
    average_accuracy: float | None
    macro_f1: float | None
    average_time: float | None
    stdout: str
    stderr: str
    command: list[str]
    return_code: int = 0
    coarse_graph_path: str | None = None
    exports_pt_dir: str | None = None
    exports_text_dir: str | None = None
    metadata_path: str | None = None
    spectral_metrics_path: str | None = None


class UGCService:
    def __init__(self, ugc_script_path: Path, python_command: list[str]) -> None:
        self.ugc_script_path = ugc_script_path
        self.python_command = python_command

    def run(
        self,
        dataset: str,
        edge_index_path: Path,
        node_feat_path: Path,
        label_path: Path,
        feature_size: int,
        num_classes: int,
        alpha: float,
        bin_width: float,
        save_coarse_dir: Path | None = None,
        model_type: str = "gcn",
        ratio: int = 50,
        epochs: int = 50,
        seed: int = 42,
        coarsen_only: bool = False,
        calculate_spectral_errors: bool = False,
        purity_threshold: float = 0.7,
        enable_similarity_weighted_features: bool = False,
        similarity_temperature: float = 5.0,
        ) -> UGCResult:
        command = self.build_command(
            dataset=dataset,
            edge_index_path=edge_index_path,
            node_feat_path=node_feat_path,
            label_path=label_path,
            feature_size=feature_size,
            num_classes=num_classes,
            alpha=alpha,
            bin_width=bin_width,
            save_coarse_dir=save_coarse_dir,
            model_type=model_type,
            ratio=ratio,
            epochs=epochs,
            seed=seed,
            coarsen_only=coarsen_only,
            calculate_spectral_errors=calculate_spectral_errors,
            purity_threshold=purity_threshold,
            enable_similarity_weighted_features=enable_similarity_weighted_features,
            similarity_temperature=similarity_temperature,
        )
        completed = run_subprocess(command, cwd=self.ugc_script_path.parent)
        return self._build_result(
            dataset=dataset,
            model_type=model_type,
            epochs=epochs,
            bin_width=bin_width,
            coarsen_only=coarsen_only,
            completed=completed,
            save_coarse_dir=save_coarse_dir,
            command=command,
        )

    def build_command(
        self,
        dataset: str,
        edge_index_path: Path,
        node_feat_path: Path,
        label_path: Path,
        feature_size: int,
        num_classes: int,
        alpha: float,
        bin_width: float,
        save_coarse_dir: Path | None = None,
        model_type: str = "gcn",
        ratio: int = 50,
        epochs: int = 50,
        seed: int = 42,
        coarsen_only: bool = False,
        calculate_spectral_errors: bool = False,
        purity_threshold: float = 0.7,
        enable_similarity_weighted_features: bool = False,
        similarity_temperature: float = 5.0,
    ) -> list[str]:
        command = self.python_command + [
            str(self.ugc_script_path),
            "--dataset",
            dataset,
            "--dataset_not_in_torch_geometric",
            "True",
            "--edge_index_path",
            str(edge_index_path),
            "--node_feat_path",
            str(node_feat_path),
            "--label_path",
            str(label_path),
            "--feature_size",
            str(feature_size),
            "--num_classes",
            str(num_classes),
            "--model_type",
            model_type,
            "--ratio",
            str(ratio),
            "--add_adj_to_node_features",
            "True",
            "--alpha",
            str(alpha),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--bin_width",
            str(bin_width),
        ]
        if save_coarse_dir is not None:
            command.extend(["--save_coarse_dir", str(save_coarse_dir)])
        if coarsen_only:
            command.extend(["--coarsen_only", "True"])
        if calculate_spectral_errors:
            command.extend(["--calculate_spectral_errors", "True"])
        command.extend(["--purity_threshold", str(purity_threshold)])
        if enable_similarity_weighted_features:
            command.extend(["--enable_similarity_weighted_features", "True"])
            command.extend(["--similarity_temperature", str(similarity_temperature)])
        return command

    def run_streaming(
        self,
        dataset: str,
        edge_index_path: Path,
        node_feat_path: Path,
        label_path: Path,
        feature_size: int,
        num_classes: int,
        alpha: float,
        bin_width: float,
        save_coarse_dir: Path | None = None,
        model_type: str = "gcn",
        ratio: int = 50,
        epochs: int = 50,
        seed: int = 42,
        coarsen_only: bool = False,
        calculate_spectral_errors: bool = False,
        purity_threshold: float = 0.7,
        enable_similarity_weighted_features: bool = False,
        similarity_temperature: float = 5.0,
        progress_callback=None,
    ) -> UGCResult:
        command = self.build_command(
            dataset=dataset,
            edge_index_path=edge_index_path,
            node_feat_path=node_feat_path,
            label_path=label_path,
            feature_size=feature_size,
            num_classes=num_classes,
            alpha=alpha,
            bin_width=bin_width,
            save_coarse_dir=save_coarse_dir,
            model_type=model_type,
            ratio=ratio,
            epochs=epochs,
            seed=seed,
            coarsen_only=coarsen_only,
            calculate_spectral_errors=calculate_spectral_errors,
            purity_threshold=purity_threshold,
            enable_similarity_weighted_features=enable_similarity_weighted_features,
            similarity_temperature=similarity_temperature,
        )
        process = subprocess.Popen(
            command,
            cwd=str(self.ugc_script_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )
        stdout_lines: list[str] = []
        if process.stdout is not None:
            for line in process.stdout:
                stdout_lines.append(line)
                if progress_callback:
                    progress_callback(line.rstrip("\n"))
        return_code = process.wait()
        completed = subprocess.CompletedProcess(
            args=command,
            returncode=return_code,
            stdout="".join(stdout_lines),
            stderr="",
        )
        return self._build_result(
            dataset=dataset,
            model_type=model_type,
            epochs=epochs,
            bin_width=bin_width,
            coarsen_only=coarsen_only,
            completed=completed,
            save_coarse_dir=save_coarse_dir,
            command=command,
        )

    def _build_result(
        self,
        dataset: str,
        model_type: str,
        epochs: int,
        bin_width: float,
        coarsen_only: bool,
        completed,
        save_coarse_dir: Path | None,
        command: list[str],
    ) -> UGCResult:
        stdout = completed.stdout
        coarse_graph_path = None
        exports_pt_dir = None
        exports_text_dir = None
        metadata_path = None
        spectral_metrics_path = None
        if save_coarse_dir is not None:
            coarse_graph_file = save_coarse_dir / "coarse_graph.json"
            pt_dir = save_coarse_dir / "exports" / "pt"
            text_dir = save_coarse_dir / "exports" / "text"
            meta_file = save_coarse_dir / "metadata.json"
            spectral_metrics_file = save_coarse_dir / "spectral_metrics.json"
            coarse_graph_path = str(coarse_graph_file) if coarse_graph_file.exists() else None
            exports_pt_dir = str(pt_dir) if pt_dir.exists() else None
            exports_text_dir = str(text_dir) if text_dir.exists() else None
            metadata_path = str(meta_file) if meta_file.exists() else None
            spectral_metrics_path = str(spectral_metrics_file) if spectral_metrics_file.exists() else None
        return UGCResult(
            dataset=dataset,
            model_type=model_type,
            epochs=epochs,
            bin_width=bin_width,
            coarsen_only=coarsen_only,
            reduction_percent=parse_float(stdout, r"Graph reduced by:\s*([\d\.]+)\s*percent"),
            supernodes=parse_int(stdout, r"We now have\s+(\d+)\s+supernode"),
            original_nodes=parse_int(stdout, r"starting nodes were:\s*(\d+)"),
            average_accuracy=parse_float(stdout, r"ave_acc:\s*([\d\.]+)"),
            macro_f1=parse_float(stdout, r"ave_macro_f1:\s*([\d\.]+)"),
            average_time=parse_float(stdout, r"ave_time:\s*([\d\.]+)"),
            stdout=stdout,
            stderr=completed.stderr,
            command=command,
            return_code=completed.returncode,
            coarse_graph_path=coarse_graph_path,
            exports_pt_dir=exports_pt_dir,
            exports_text_dir=exports_text_dir,
            metadata_path=metadata_path,
            spectral_metrics_path=spectral_metrics_path,
        )
