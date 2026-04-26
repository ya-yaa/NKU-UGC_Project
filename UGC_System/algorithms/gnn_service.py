from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GNNOption:
    key: str
    label: str


class GNNService:
    def list_builtin_fast_datasets(self) -> dict[str, dict]:
        return {
            "cora": {
                "allowed_ratios": [10, 30, 50, 70, 90],
                "strategy_locked": True,
                "skip_preprocess": True,
                "skip_calibration": True,
            },
            "citeseer": {
                "allowed_ratios": [10, 30, 50, 70, 90],
                "strategy_locked": True,
                "skip_preprocess": True,
                "skip_calibration": True,
            },
        }

    def list_supported_models(self) -> list[GNNOption]:
        return [
            GNNOption("gcn", "GCN"),
            GNNOption("sage", "GraphSAGE"),
            GNNOption("gat", "GAT"),
            GNNOption("gin", "GIN"),
            GNNOption("ugc", "APPNP"),
            GNNOption("3wl", "3WL"),
        ]

    def list_strategies(self) -> list[GNNOption]:
        return [
            GNNOption("v1", "V1 First Label"),
            GNNOption("v2", "V2 Set Intersection"),
            GNNOption("v3", "V3 Position Match"),
            GNNOption("v4", "V4 Jaccard Threshold"),
        ]
