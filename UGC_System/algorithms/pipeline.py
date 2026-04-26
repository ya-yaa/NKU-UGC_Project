from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from config import DATA_DIR, OUTPUTS_DIR, SCRIPTS_DIR

from .analysis_service import AnalysisResult, AnalysisService
from .alpha_service import AlphaResult, AlphaService
from .calibrate_service import CalibrateService, CalibrationResult
from .cleaning_service import CleaningResult, CleaningService
from .preprocess_service import PreprocessResult, PreprocessService
from .ugc_service import UGCResult, UGCService
from .utils import ensure_dir, read_json, resolve_python_command, timestamp_slug, write_json


BUILTIN_UGC_DATASETS = {
    "cora": {
        "allowed_ratios": {10, 30, 50, 70, 90},
        "bin_widths": {10: 0.0006, 30: 0.0021, 50: 0.004, 70: 0.0125, 90: 0.052},
        "alpha": 0.19,
    },
    "citeseer": {
        "allowed_ratios": {10, 30, 50, 70, 90},
        "bin_widths": {10: 0.00073, 30: 0.0029, 50: 0.0065, 70: 0.0148, 90: 0.0625},
        "alpha": 0.26,
    },
}


def parse_ugc_progress_line(line: str) -> dict | None:
    marker = "UGC_STAGE:"
    if marker not in line:
        return None
    substage = line.split(marker, 1)[1].strip()
    if not substage:
        return None
    return {"status": "running", "substage": substage}


@dataclass
class PipelineResult:
    run_id: str
    dataset: str
    strategy: str
    data_root: str
    target_ratio: float
    seed: int
    purity_threshold: float
    enable_similarity_weighted_features: bool
    similarity_temperature: float
    analysis: AnalysisResult
    cleaning: CleaningResult
    alpha: AlphaResult
    preprocess: PreprocessResult
    calibration: CalibrationResult
    ugc: UGCResult
    coarsening: dict | None
    output_dir: str


class UGCPipeline:
    def __init__(self) -> None:
        python_command = resolve_python_command()
        self.analysis_service = AnalysisService()
        self.cleaning_service = CleaningService()
        self.alpha_service = AlphaService(SCRIPTS_DIR)
        self.preprocess_service = PreprocessService(SCRIPTS_DIR)
        self.calibrate_service = CalibrateService(SCRIPTS_DIR / "UGC.py", python_command)
        self.ugc_service = UGCService(SCRIPTS_DIR / "UGC.py", python_command)

    def run(
        self,
        dataset: str = "facebook",
        strategy: str = "v4",
        jaccard_threshold: float = 0.5,
        out_subdir: str | None = None,
        target_ratio: float = 50.0,
        data_root: Path | None = None,
        force_undirected: bool | None = None,
        model_type: str = "gcn",
        train_epochs: int = 1,
        calibrate_iters: int = 6,
        seed: int = 42,
        calculate_spectral_errors: bool = False,
        purity_threshold: float = 0.7,
        enable_similarity_weighted_features: bool = False,
        similarity_temperature: float = 5.0,
        run_id: str | None = None,
        stage_callback: Callable[[str, dict], None] | None = None,
    ) -> PipelineResult:
        builtin_config = BUILTIN_UGC_DATASETS.get(dataset.lower())
        if strategy not in {"v1", "v2", "v3", "v4"}:
            raise ValueError(f"Unsupported strategy: {strategy}")
        if builtin_config and int(target_ratio) not in builtin_config["allowed_ratios"]:
            allowed = ", ".join(str(value) for value in sorted(builtin_config["allowed_ratios"]))
            raise ValueError(f"{dataset} only supports target ratios: {allowed}")
        if out_subdir is None:
            out_subdir = "builtin_pt" if builtin_config else f"ugc_{strategy}"
        if data_root is None:
            data_root = DATA_DIR
        run_id = run_id or f"{dataset}_{strategy}_{timestamp_slug()}"
        output_dir = ensure_dir(OUTPUTS_DIR / run_id)

        if stage_callback:
            stage_callback("analysis", {"status": "running"})
        analysis_result = self.analysis_service.analyze(dataset=dataset, data_root=data_root)
        if stage_callback:
            stage_callback(
                "analysis",
                {
                    "status": "completed",
                    "num_nodes": analysis_result.num_nodes,
                    "num_edges": analysis_result.num_edges,
                    "feature_dim": analysis_result.feature_dim,
                    "label_type": analysis_result.label_type,
                    "label_dim": analysis_result.label_dim,
                    "avg_degree": analysis_result.avg_degree,
                    "max_degree": analysis_result.max_degree,
                    "source": analysis_result.source,
                },
            )

        if stage_callback:
            stage_callback("cleaning", {"status": "running"})
        cleaning_root = output_dir / "_working_data"
        cleaning_result = self.cleaning_service.clean(dataset=dataset, source_root=data_root, cleaned_root=cleaning_root)
        effective_data_root = Path(cleaning_result.cleaned_data_root)
        if stage_callback:
            stage_callback(
                "cleaning",
                {
                    "status": "completed",
                    "num_nodes_before": cleaning_result.num_nodes_before,
                    "num_nodes_after": cleaning_result.num_nodes_after,
                    "num_edges_before": cleaning_result.num_edges_before,
                    "num_edges_after": cleaning_result.num_edges_after,
                    "removed_unlabeled_nodes": cleaning_result.removed_unlabeled_nodes,
                    "removed_duplicate_edges": cleaning_result.removed_duplicate_edges,
                    "skipped": cleaning_result.skipped,
                    "reason": cleaning_result.reason,
                },
            )

        if builtin_config:
            if stage_callback:
                stage_callback("alpha", {"status": "completed", "alpha": builtin_config["alpha"], "mode": "builtin"})
            alpha_result = AlphaResult(
                dataset=dataset,
                strategy="builtin",
                alpha=float(builtin_config["alpha"]),
                labeled_edges=0,
                total_edges=0,
                coverage_percent=0.0,
                jaccard_threshold=None,
            )

            if stage_callback:
                stage_callback("preprocess", {"status": "running", "mode": "builtin"})
            preprocess_result = self.preprocess_service.load_builtin_processed_dataset(
                data_root=effective_data_root,
                dataset=dataset,
                out_subdir=out_subdir,
            )
            if stage_callback:
                stage_callback(
                    "preprocess",
                    {
                        "status": "completed",
                        "num_nodes": preprocess_result.num_nodes,
                        "num_classes": preprocess_result.num_classes,
                        "edge_index_path": preprocess_result.edge_index_path,
                        "node_feat_path": preprocess_result.node_feat_path,
                        "label_path": preprocess_result.label_path,
                        "mode": "builtin",
                    },
                )

            if stage_callback:
                stage_callback(
                    "calibration",
                    {
                        "status": "completed",
                        "mode": "builtin",
                        "skipped": True,
                        "bin_width": float(builtin_config["bin_widths"][int(target_ratio)]),
                    },
                )
            calibration_result = CalibrationResult(
                dataset=dataset,
                target_ratio=target_ratio,
                recommended_bin_width=float(builtin_config["bin_widths"][int(target_ratio)]),
                actual_ratio=None,
                iterations=[],
                stdout="Skipped calibration for built-in UGC dataset; using predefined bin width.",
                stderr="",
                return_code=0,
            )
        else:
            if stage_callback:
                stage_callback("alpha", {"status": "running"})

            alpha_result = self.alpha_service.calculate(strategy, effective_data_root, dataset, jaccard_threshold)
            if stage_callback:
                stage_callback(
                    "alpha",
                    {
                        "status": "completed",
                        "alpha": alpha_result.alpha,
                        "coverage_percent": alpha_result.coverage_percent,
                    },
                )

            if stage_callback:
                stage_callback("preprocess", {"status": "running"})
            preprocess_result = self.preprocess_service.run(
                strategy=strategy,
                data_root=effective_data_root,
                dataset=dataset,
                out_subdir=out_subdir,
                jaccard_threshold=jaccard_threshold,
                force_undirected=force_undirected,
            )
            if stage_callback:
                stage_callback(
                    "preprocess",
                    {
                        "status": "completed",
                        "num_nodes": preprocess_result.num_nodes,
                        "num_classes": preprocess_result.num_classes,
                        "edge_index_path": preprocess_result.edge_index_path,
                        "node_feat_path": preprocess_result.node_feat_path,
                        "label_path": preprocess_result.label_path,
                    },
                )

            if stage_callback:
                stage_callback("calibration", {"status": "running"})
            calibration_result = self.calibrate_service.calibrate(
                dataset=dataset,
                edge_index_path=Path(preprocess_result.edge_index_path),
                node_feat_path=Path(preprocess_result.node_feat_path),
                label_path=Path(preprocess_result.label_path),
                feature_size=preprocess_result.feature_size,
                num_classes=preprocess_result.num_classes,
                alpha=alpha_result.alpha,
                model_type=model_type,
                target_ratio=target_ratio,
                epochs=1,
                iterations=calibrate_iters,
                seed=seed,
                purity_threshold=purity_threshold,
                progress_callback=(
                    (lambda payload: stage_callback("calibration_progress", payload))
                    if stage_callback
                    else None
                ),
            )
            if calibration_result.return_code != 0 or calibration_result.actual_ratio is None:
                error_message = calibration_result.stderr.strip() or calibration_result.stdout.strip() or "Bin width calibration failed."
                raise RuntimeError(f"桶宽标定失败：{error_message}")
            if stage_callback:
                stage_callback(
                    "calibration",
                    {
                        "status": "completed",
                        "bin_width": calibration_result.recommended_bin_width,
                        "actual_ratio": calibration_result.actual_ratio,
                        "iterations": calibration_result.iterations,
                    },
                )

        if stage_callback:
            stage_callback("ugc", {"status": "running"})

        def handle_ugc_progress(line: str) -> None:
            if not stage_callback:
                return
            payload = parse_ugc_progress_line(line)
            if payload:
                stage_callback("ugc_progress", payload)

        ugc_result = self.ugc_service.run_streaming(
            dataset=dataset,
            edge_index_path=Path(preprocess_result.edge_index_path),
            node_feat_path=Path(preprocess_result.node_feat_path),
            label_path=Path(preprocess_result.label_path),
            feature_size=preprocess_result.feature_size,
            num_classes=preprocess_result.num_classes,
            alpha=alpha_result.alpha,
            bin_width=calibration_result.recommended_bin_width,
            save_coarse_dir=output_dir,
            model_type=model_type,
            ratio=int(target_ratio),
            epochs=train_epochs,
            seed=seed,
            coarsen_only=True,
            calculate_spectral_errors=calculate_spectral_errors,
            purity_threshold=purity_threshold,
            enable_similarity_weighted_features=enable_similarity_weighted_features,
            similarity_temperature=similarity_temperature,
            progress_callback=handle_ugc_progress,
        )
        if ugc_result.return_code != 0 or ugc_result.reduction_percent is None:
            error_message = ugc_result.stderr.strip() or ugc_result.stdout.strip() or "UGC coarsening failed."
            raise RuntimeError(f"UGC 粗化失败：{error_message}")
        coarse_meta = None
        if ugc_result.coarse_graph_path:
            coarse_graph = read_json(Path(ugc_result.coarse_graph_path), default={}) or {}
            coarse_meta = coarse_graph.get("meta")
        if stage_callback:
            stage_callback(
                "ugc",
                {
                    "status": "completed",
                    "original_nodes": ugc_result.original_nodes,
                    "compressed_nodes": ugc_result.supernodes,
                    "reduction_percent": ugc_result.reduction_percent,
                    "average_accuracy": ugc_result.average_accuracy,
                    "coarsen_only": ugc_result.coarsen_only,
                    "coarse_graph_path": ugc_result.coarse_graph_path,
                },
            )
        result = PipelineResult(
            run_id=run_id,
            dataset=dataset,
            strategy=strategy,
            data_root=str(data_root),
            target_ratio=target_ratio,
            seed=seed,
            purity_threshold=purity_threshold,
            enable_similarity_weighted_features=enable_similarity_weighted_features,
            similarity_temperature=similarity_temperature,
            analysis=analysis_result,
            cleaning=cleaning_result,
            alpha=alpha_result,
            preprocess=preprocess_result,
            calibration=calibration_result,
            ugc=ugc_result,
            coarsening=coarse_meta,
            output_dir=str(output_dir),
        )
        write_json(output_dir / "summary.json", asdict(result))
        (output_dir / "ugc_stdout.log").write_text(ugc_result.stdout, encoding="utf-8")
        (output_dir / "ugc_stderr.log").write_text(ugc_result.stderr, encoding="utf-8")
        if stage_callback:
            stage_callback(
                "completed",
                {
                    "status": "completed",
                    "output_dir": str(output_dir),
                    "run_id": run_id,
                },
            )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified UGC pipeline.")
    parser.add_argument("--dataset", default="facebook")
    parser.add_argument("--strategy", default="v4")
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--out-subdir", default=None)
    parser.add_argument("--target-ratio", type=float, default=50.0)
    parser.add_argument("--model-type", default="gcn")
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--calibrate-iters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calculate-spectral-errors", action="store_true")
    parser.add_argument("--purity-threshold", type=float, default=0.7)
    parser.add_argument("--enable-similarity-weighted-features", action="store_true")
    parser.add_argument("--similarity-temperature", type=float, default=5.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = UGCPipeline()
    result = pipeline.run(
        dataset=args.dataset,
        strategy=args.strategy,
        jaccard_threshold=args.jaccard_threshold,
        out_subdir=args.out_subdir,
        target_ratio=args.target_ratio,
        model_type=args.model_type,
        train_epochs=args.train_epochs,
        calibrate_iters=args.calibrate_iters,
        seed=args.seed,
        calculate_spectral_errors=args.calculate_spectral_errors,
        purity_threshold=args.purity_threshold,
        enable_similarity_weighted_features=args.enable_similarity_weighted_features,
        similarity_temperature=args.similarity_temperature,
    )
    print(f"Run ID: {result.run_id}")
    print(f"Output dir: {result.output_dir}")
    print(f"Alpha: {result.alpha.alpha:.4f}")
    print(f"Bin width: {result.calibration.recommended_bin_width:.6f}")
    if result.ugc.reduction_percent is not None:
        print(f"Reduction: {result.ugc.reduction_percent:.2f}%")
    if result.ugc.average_accuracy is not None:
        print(f"Average accuracy: {result.ugc.average_accuracy:.4f}")


if __name__ == "__main__":
    main()
