from __future__ import annotations

from .mysql_repository import MySQLExperimentRepository
from .repository import ExperimentRepository


def get_experiment_repository() -> ExperimentRepository:
    """Return the active MySQL-backed experiment repository."""

    return MySQLExperimentRepository()
