from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import parse_float, run_subprocess


@dataclass
class CalibrationResult:
    dataset: str
    target_ratio: float
    recommended_bin_width: float
    actual_ratio: float | None
    iterations: list[dict]
    stdout: str
    stderr: str
    return_code: int = 0


class CalibrateService:
    def __init__(self, ugc_script_path: Path, python_command: list[str]) -> None:
        self.ugc_script_path = ugc_script_path
        self.python_command = python_command

    def _run_and_get_ratio(self, args_list: list[str]) -> tuple[float | None, str, str, int]:
        completed = run_subprocess(self.python_command + [str(self.ugc_script_path)] + args_list, cwd=self.ugc_script_path.parent)
        ratio = parse_float(completed.stdout, r"Graph reduced by:\s*([\d\.]+)\s*percent")
        return ratio, completed.stdout, completed.stderr, completed.returncode

    def calibrate(
        self,
        dataset: str,
        edge_index_path: Path,
        node_feat_path: Path,
        label_path: Path,
        feature_size: int,
        num_classes: int,
        alpha: float,
        model_type: str = "gcn",
        target_ratio: float = 50.0,
        epochs: int = 1,
        iterations: int = 8,
        low: float = 0.00001,
        high: float = 0.5,
        seed: int = 42,
        purity_threshold: float = 0.7,
        progress_callback=None,
    ) -> CalibrationResult:
        base_args = [
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
            str(int(target_ratio)),
            "--add_adj_to_node_features",
            "True",
            "--alpha",
            str(alpha),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--coarsen_only",
            "True",
            "--purity_threshold",
            str(purity_threshold),
        ]

        best_width = low
        best_diff = float("inf")
        final_ratio = None
        final_stdout = ""
        final_stderr = ""
        final_return_code = 0
        iteration_logs: list[dict] = []

        current_low = low
        current_high = high
        for index in range(1, iterations + 1):
            mid = (current_low + current_high) / 2.0
            if progress_callback:
                progress_callback(
                    {
                        "iteration": index,
                        "iterations": iterations,
                        "bin_width": mid,
                        "low": current_low,
                        "high": current_high,
                        "status": "running",
                    }
                )
            actual_ratio, stdout, stderr, return_code = self._run_and_get_ratio(base_args + ["--bin_width", str(mid)])
            if actual_ratio is None:
                final_stdout = stdout
                final_stderr = stderr
                final_return_code = return_code
                if progress_callback:
                    progress_callback(
                        {
                            "iteration": index,
                            "iterations": iterations,
                            "bin_width": mid,
                            "status": "failed",
                            "stderr": stderr,
                            "stdout": stdout,
                        }
                    )
                break

            diff = abs(actual_ratio - target_ratio)
            iteration_logs.append(
                {
                    "iteration": index,
                    "bin_width": mid,
                    "actual_ratio": actual_ratio,
                    "diff": diff,
                }
            )
            final_ratio = actual_ratio
            final_stdout = stdout
            final_stderr = stderr
            final_return_code = return_code
            if progress_callback:
                progress_callback(
                    {
                        "iteration": index,
                        "iterations": iterations,
                        "bin_width": mid,
                        "actual_ratio": actual_ratio,
                        "diff": diff,
                        "status": "completed",
                    }
                )

            if diff < best_diff:
                best_diff = diff
                best_width = mid

            if actual_ratio < target_ratio:
                current_low = mid
            else:
                current_high = mid

        return CalibrationResult(
            dataset=dataset,
            target_ratio=target_ratio,
            recommended_bin_width=best_width,
            actual_ratio=final_ratio,
            iterations=iteration_logs,
            stdout=final_stdout,
            stderr=final_stderr,
            return_code=final_return_code,
        )
