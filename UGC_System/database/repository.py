from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass
class ExperimentCreatePayload:
    run_id: str
    dataset_name: str
    label_type: str
    label_strategy: str
    target_ratio: float
    use_weighted_feature: bool
    gamma: float | None
    alpha: float | None
    bin_width: float | None
    actual_ratio: float | None
    avg_eigen_error: float | None
    de_error: float | None
    avg_purity: float | None
    low_purity_ratio: float | None
    status: str
    result_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentUpdatePayload:
    label_type: str | None = None
    label_strategy: str | None = None
    target_ratio: float | None = None
    use_weighted_feature: bool | None = None
    gamma: float | None = None
    alpha: float | None = None
    bin_width: float | None = None
    actual_ratio: float | None = None
    avg_eigen_error: float | None = None
    de_error: float | None = None
    avg_purity: float | None = None
    low_purity_ratio: float | None = None
    status: str | None = None
    result_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class GNNTestCreatePayload:
    experiment_id: int
    model_name: str
    epochs: int
    accuracy: float | None
    train_time: float | None
    status: str
    log_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GNNTestUpdatePayload:
    model_name: str | None = None
    epochs: int | None = None
    accuracy: float | None = None
    train_time: float | None = None
    status: str | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class ExperimentRepository(Protocol):
    def create_experiment(self, payload: ExperimentCreatePayload) -> int | None:
        ...

    def update_experiment(self, experiment_id: int, payload: ExperimentUpdatePayload) -> None:
        ...

    def create_gnn_test_result(self, payload: GNNTestCreatePayload) -> int | None:
        ...

    def update_gnn_test_result(self, test_id: int, payload: GNNTestUpdatePayload) -> None:
        ...

    def list_gnn_test_results(self, experiment_id: int) -> list[dict[str, Any]]:
        ...


class NullExperimentRepository:
    """No-op repository used until a real database adapter is connected."""

    def create_experiment(self, payload: ExperimentCreatePayload) -> int | None:
        return None

    def update_experiment(self, experiment_id: int, payload: ExperimentUpdatePayload) -> None:
        return None

    def create_gnn_test_result(self, payload: GNNTestCreatePayload) -> int | None:
        return None

    def update_gnn_test_result(self, test_id: int, payload: GNNTestUpdatePayload) -> None:
        return None

    def list_gnn_test_results(self, experiment_id: int) -> list[dict[str, Any]]:
        return []
