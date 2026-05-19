from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from .repository import (
    ExperimentCreatePayload,
    ExperimentRepository,
    ExperimentUpdatePayload,
    GNNTestCreatePayload,
    GNNTestUpdatePayload,
)


EXPERIMENT_INSERT_FIELDS = [
    "dataset_name",
    "label_type",
    "label_strategy",
    "target_ratio",
    "use_weighted_feature",
    "gamma",
    "alpha",
    "bin_width",
    "actual_ratio",
    "avg_eigen_error",
    "de_error",
    "avg_purity",
    "low_purity_ratio",
    "status",
    "result_path",
]

EXPERIMENT_UPDATE_FIELDS = [
    "label_type",
    "label_strategy",
    "target_ratio",
    "use_weighted_feature",
    "gamma",
    "alpha",
    "bin_width",
    "actual_ratio",
    "avg_eigen_error",
    "de_error",
    "avg_purity",
    "low_purity_ratio",
    "status",
    "result_path",
]

GNN_TEST_INSERT_FIELDS = [
    "experiment_id",
    "model_name",
    "epochs",
    "accuracy",
    "train_time",
    "status",
    "log_path",
]

GNN_TEST_UPDATE_FIELDS = [
    "model_name",
    "epochs",
    "accuracy",
    "train_time",
    "status",
    "log_path",
]


class MySQLExperimentRepository(ExperimentRepository):
    """MySQL implementation for experiment and GNN test result storage."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        *,
        charset: str = "utf8mb4",
        connect_timeout: int = 10,
        autocommit: bool = False,
        connection_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.host = host or os.environ.get("UGC_DB_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("UGC_DB_PORT", "3306"))
        self.user = user or os.environ.get("UGC_DB_USER", "root")
        self.password = password or os.environ.get("UGC_DB_PASSWORD", "Root12345!")
        self.database = database or os.environ.get("UGC_DB_NAME", "ugc_system")
        self.charset = charset
        self.connect_timeout = connect_timeout
        self.autocommit = autocommit
        self.connection_factory = connection_factory or pymysql.connect
        self.connection = None

    def _connect(self):
        return self.connection_factory(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            cursorclass=DictCursor,
            autocommit=self.autocommit,
            connect_timeout=self.connect_timeout,
        )

    def _get_connection(self):
        if self.connection is None:
            self.connection = self._connect()
            return self.connection
        try:
            self.connection.ping(reconnect=True)
        except Exception:
            self.connection = self._connect()
        return self.connection

    def close(self) -> None:
        if self.connection is None:
            return
        try:
            self.connection.close()
        finally:
            self.connection = None

    def _filter_payload(self, payload_dict: dict[str, Any], allowed_fields: list[str], *, keep_none: bool) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field in allowed_fields:
            if field not in payload_dict:
                continue
            value = payload_dict[field]
            if value is None and not keep_none:
                continue
            data[field] = value
        return data

    def _execute_write(self, sql: str, params: dict[str, Any] | tuple[Any, ...]) -> int | None:
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                lastrowid = getattr(cursor, "lastrowid", None)
            if not self.autocommit:
                connection.commit()
            return lastrowid
        except Exception:
            if not self.autocommit:
                connection.rollback()
            raise

    def _execute_fetchall(self, sql: str, params: dict[str, Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
        connection = self._get_connection()
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def create_experiment(self, payload: ExperimentCreatePayload) -> int | None:
        data = self._filter_payload(payload.to_dict(), EXPERIMENT_INSERT_FIELDS, keep_none=True)
        columns = ", ".join(data.keys())
        values = ", ".join(f"%({field})s" for field in data.keys())
        sql = f"""
        INSERT INTO experiment (
            {columns}
        ) VALUES (
            {values}
        )
        """
        return self._execute_write(sql, data)

    def update_experiment(self, experiment_id: int, payload: ExperimentUpdatePayload) -> None:
        data = self._filter_payload(payload.to_dict(), EXPERIMENT_UPDATE_FIELDS, keep_none=False)
        if not data:
            return
        assignments = ", ".join(f"{field} = %({field})s" for field in data.keys())
        sql = f"""
        UPDATE experiment
        SET {assignments}, updated_at = NOW()
        WHERE id = %(id)s
        """
        data["id"] = experiment_id
        self._execute_write(sql, data)

    def create_gnn_test_result(self, payload: GNNTestCreatePayload) -> int | None:
        data = self._filter_payload(payload.to_dict(), GNN_TEST_INSERT_FIELDS, keep_none=True)
        columns = ", ".join(data.keys())
        values = ", ".join(f"%({field})s" for field in data.keys())
        sql = f"""
        INSERT INTO gnn_test_result (
            {columns}
        ) VALUES (
            {values}
        )
        """
        return self._execute_write(sql, data)

    def update_gnn_test_result(self, test_id: int, payload: GNNTestUpdatePayload) -> None:
        data = self._filter_payload(payload.to_dict(), GNN_TEST_UPDATE_FIELDS, keep_none=False)
        if not data:
            return
        assignments = ", ".join(f"{field} = %({field})s" for field in data.keys())
        sql = f"""
        UPDATE gnn_test_result
        SET {assignments}
        WHERE id = %(id)s
        """
        data["id"] = test_id
        self._execute_write(sql, data)

    def list_gnn_test_results(self, experiment_id: int) -> list[dict[str, Any]]:
        sql = """
        SELECT
            id,
            experiment_id,
            model_name,
            epochs,
            accuracy,
            train_time,
            status,
            log_path,
            created_at
        FROM gnn_test_result
        WHERE experiment_id = %s
        ORDER BY created_at DESC, id DESC
        """
        return self._execute_fetchall(sql, (experiment_id,))
