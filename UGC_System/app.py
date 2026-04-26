from __future__ import annotations

import shutil
import threading
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from algorithms.gnn_service import GNNService
from algorithms.pipeline import UGCPipeline
from algorithms.utils import ensure_dir, read_json, slugify, timestamp_slug, write_json
from config import DATA_DIR, OUTPUTS_DIR, UPLOADS_DIR


app = Flask(__name__)
pipeline = UGCPipeline()
gnn_service = GNNService()


def render_upload_config(entry_mode: str, error: str | None = None):
    if entry_mode not in {"builtin", "upload"}:
        abort(404)
    return render_template(
        "upload.html",
        entry_mode=entry_mode,
        models=gnn_service.list_supported_models(),
        strategies=gnn_service.list_strategies(),
        datasets=list_dataset_choices(),
        builtin_fast_datasets=gnn_service.list_builtin_fast_datasets(),
        error=error,
    )


def experiment_dir(run_id: str) -> Path:
    return ensure_dir(OUTPUTS_DIR / run_id)


def state_path(run_id: str) -> Path:
    return experiment_dir(run_id) / "state.json"


def summary_path(run_id: str) -> Path:
    return experiment_dir(run_id) / "summary.json"


def gnn_state_path(run_id: str) -> Path:
    return experiment_dir(run_id) / "gnn_test_state.json"


def list_dataset_choices() -> list[str]:
    datasets = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    return datasets or ["facebook"]


def make_stage_cards(state: dict) -> list[dict]:
    stage_names = {
        "analysis": "数据分析",
        "cleaning": "数据清洗",
        "alpha": "计算 Alpha",
        "preprocess": "数据预处理",
        "calibration": "标定 Bin Width",
        "ugc": "运行 UGC 粗化",
    }
    cards = []
    stage_results = state.get("stage_results", {})
    current_stage = state.get("current_stage")
    for key in ["analysis", "cleaning", "alpha", "preprocess", "calibration", "ugc"]:
        payload = stage_results.get(key, {})
        status = payload.get("status", "pending")
        if current_stage == key and status == "pending":
            status = "running"
        cards.append({"key": key, "label": stage_names[key], "status": status, "payload": payload})
    return cards


def build_graph_preview(run_id: str) -> dict:
    coarse_path = experiment_dir(run_id) / "coarse_graph.json"
    graph_preview = read_json(
        coarse_path,
        default={"nodes": [], "edges": [], "details": {}, "meta": {}, "note": "暂无真实 coarse graph 数据。"},
    )
    return graph_preview


def build_original_mapping(summary: dict, graph_preview: dict) -> dict:
    mapping_path = experiment_dir(summary["run_id"]) / "mapping.json"
    saved_mapping = read_json(mapping_path, default=None)
    if not saved_mapping:
        raise FileNotFoundError(f"Missing mapping.json for run {summary['run_id']}")
    return saved_mapping


@lru_cache(maxsize=8)
def load_mapping_payload(run_id: str) -> dict:
    mapping_path = experiment_dir(run_id) / "mapping.json"
    saved_mapping = read_json(mapping_path, default=None)
    if not saved_mapping:
        raise FileNotFoundError(f"Missing mapping.json for run {run_id}")
    return saved_mapping


def list_gnn_tests(run_id: str) -> list[dict]:
    test_dir = experiment_dir(run_id) / "gnn_tests"
    if not test_dir.exists():
        return []
    results = []
    for path in sorted(test_dir.glob("*.json"), reverse=True):
        payload = read_json(path, {})
        payload["filename"] = path.name
        results.append(payload)
    return results


def export_dir(run_id: str, export_kind: str) -> Path:
    return experiment_dir(run_id) / "exports" / export_kind


def save_state(run_id: str, payload: dict) -> None:
    write_json(state_path(run_id), payload)


def load_state(run_id: str) -> dict:
    return read_json(state_path(run_id), default={}) or {}


def save_gnn_state(run_id: str, payload: dict) -> None:
    write_json(gnn_state_path(run_id), payload)


def load_gnn_state(run_id: str) -> dict:
    return read_json(gnn_state_path(run_id), default={}) or {}


def create_initial_state(run_id: str, form_data: dict, dataset: str, dataset_display_name: str, data_root: str, entry_mode: str) -> dict:
    return {
        "run_id": run_id,
        "status": "pending",
        "current_stage": "queued",
        "execution_started": False,
        "dataset": dataset,
        "dataset_display_name": dataset_display_name,
        "entry_mode": entry_mode,
        "data_root": data_root,
        "form_data": form_data,
        "stage_results": {},
        "log_lines": ["实验已创建，等待运行。"],
        "error": None,
    }


def parse_bool_or_none(raw_value: str | None) -> bool | None:
    if raw_value == "true":
        return True
    if raw_value == "false":
        return False
    return None


def execute_experiment(run_id: str) -> None:
    state = load_state(run_id)
    if not state or state.get("execution_started"):
        return

    state["execution_started"] = True
    state["status"] = "running"
    state["current_stage"] = "analysis"
    state.setdefault("log_lines", []).append("开始执行统一流程。")
    save_state(run_id, state)

    def update_stage(stage: str, payload: dict) -> None:
        latest_state = load_state(run_id)
        if stage == "calibration_progress":
            latest_state["current_stage"] = "calibration"
        elif stage == "ugc_progress":
            latest_state["current_stage"] = "ugc"
        else:
            latest_state["current_stage"] = stage
        latest_state.setdefault("stage_results", {})[stage] = payload
        latest_state.setdefault("log_lines", []).append(f"[{stage}] {payload}")
        if stage == "completed":
            latest_state["status"] = "completed"
        save_state(run_id, latest_state)

    try:
        form_data = state["form_data"]
        result = pipeline.run(
            dataset=state["dataset"],
            strategy=form_data["strategy"],
            jaccard_threshold=form_data["jaccard_threshold"],
            out_subdir=f"ugc_{form_data['strategy']}",
            target_ratio=form_data["target_ratio"],
            data_root=Path(state["data_root"]),
            force_undirected=parse_bool_or_none(form_data["undirected"]),
            model_type=form_data["model_type"],
            train_epochs=form_data["train_epochs"],
            calibrate_iters=form_data["calibrate_iters"],
            calculate_spectral_errors=form_data.get("calculate_spectral_errors", False),
            purity_threshold=form_data.get("purity_threshold", 0.7),
            enable_similarity_weighted_features=form_data.get("enable_similarity_weighted_features", False),
            similarity_temperature=form_data.get("similarity_temperature", 5.0),
            run_id=run_id,
            stage_callback=update_stage,
        )
        summary_file = summary_path(run_id)
        summary_payload = read_json(summary_file, default={}) or {}
        summary_payload["dataset_display_name"] = state.get("dataset_display_name", result.dataset)
        summary_payload["entry_mode"] = state.get("entry_mode")
        write_json(summary_file, summary_payload)
    except Exception as exc:
        failed_state = load_state(run_id)
        failed_state["status"] = "failed"
        failed_state["error"] = str(exc)
        failed_state.setdefault("log_lines", []).append(f"流程失败: {exc}")
        save_state(run_id, failed_state)


def start_experiment_in_background(run_id: str) -> None:
    worker = threading.Thread(target=execute_experiment, args=(run_id,), daemon=True)
    worker.start()


def execute_gnn_test(run_id: str, model_type: str, epochs: int) -> None:
    summary = read_json(summary_path(run_id), default=None)
    if not summary:
        save_gnn_state(run_id, {"status": "failed", "error": "Missing experiment summary."})
        return

    state = {
        "status": "running",
        "model_type": model_type,
        "epochs": epochs,
        "current_epoch": 0,
        "latest_line": "准备开始训练...",
        "error": None,
    }
    save_gnn_state(run_id, state)

    def handle_progress(line: str) -> None:
        latest = load_gnn_state(run_id)
        latest["latest_line"] = line
        if line.startswith("EPOCH_PROGRESS:"):
            try:
                progress = line.split(":", 1)[1]
                current, total = progress.split("/", 1)
                latest["current_epoch"] = int(current)
                latest["epochs"] = int(total)
            except Exception:
                pass
        save_gnn_state(run_id, latest)

    try:
        ugc_result = pipeline.ugc_service.run_streaming(
            dataset=summary["dataset"],
            edge_index_path=Path(summary["preprocess"]["edge_index_path"]),
            node_feat_path=Path(summary["preprocess"]["node_feat_path"]),
            label_path=Path(summary["preprocess"]["label_path"]),
            feature_size=int(summary["preprocess"]["feature_size"]),
            num_classes=int(summary["preprocess"]["num_classes"]),
            alpha=float(summary["alpha"]["alpha"]),
            bin_width=float(summary["calibration"]["recommended_bin_width"]),
            model_type=model_type,
            ratio=int(summary["target_ratio"]),
            epochs=epochs,
            purity_threshold=float(summary.get("purity_threshold", 0.7)),
            enable_similarity_weighted_features=bool(summary.get("enable_similarity_weighted_features", False)),
            similarity_temperature=float(summary.get("similarity_temperature", 5.0)),
            progress_callback=handle_progress,
        )
        if ugc_result.return_code != 0:
            raise RuntimeError(ugc_result.stderr.strip() or ugc_result.stdout.strip() or "GNN test failed.")
        test_payload = {
            "test_id": f"gnn_{timestamp_slug()}",
            "model_type": model_type,
            "epochs": epochs,
            "result": asdict(ugc_result),
        }
        test_dir = ensure_dir(experiment_dir(run_id) / "gnn_tests")
        write_json(test_dir / f"{test_payload['test_id']}.json", test_payload)
        save_gnn_state(
            run_id,
            {
                "status": "completed",
                "model_type": model_type,
                "epochs": epochs,
                "current_epoch": epochs,
                "latest_line": "训练完成，正在刷新结果...",
                "error": None,
            },
        )
    except Exception as exc:
        failed = load_gnn_state(run_id)
        failed["status"] = "failed"
        failed["error"] = str(exc)
        save_gnn_state(run_id, failed)


def start_gnn_test_in_background(run_id: str, model_type: str, epochs: int) -> None:
    worker = threading.Thread(target=execute_gnn_test, args=(run_id, model_type, epochs), daemon=True)
    worker.start()


def prepare_uploaded_dataset(run_id: str, entry_mode: str):
    edge_file = request.files.get("edge_file")
    label_file = request.files.get("label_file")
    feat_file = request.files.get("feature_file")

    if entry_mode == "builtin":
        dataset = request.form.get("dataset", "facebook").strip() or "facebook"
        return dataset, dataset, DATA_DIR

    dataset_display_name = request.form.get("dataset_name", "").strip()
    if not dataset_display_name:
        raise ValueError("上传个人数据集时，请先填写数据集名称。")
    if not all(file and file.filename for file in [edge_file, label_file, feat_file]):
        raise ValueError("上传个人数据集时，需要同时提供边集、标签和特征三个文件。")

    dataset_name = f"upload_{slugify(dataset_display_name, fallback=run_id)}_{run_id}"
    raw_dir = ensure_dir(UPLOADS_DIR / dataset_name / "raw")
    edge_file.save(raw_dir / "edgelist.txt")
    label_file.save(raw_dir / "labels.txt")
    feature_name = secure_filename(feat_file.filename) or "attrs.npz"
    if Path(feature_name).suffix.lower() != ".npz":
        raise ValueError("特征文件当前只支持 .npz 格式。")
    feat_file.save(raw_dir / "attrs.npz")
    return dataset_name, dataset_display_name, UPLOADS_DIR


@app.route("/", methods=["GET"])
def cover_page():
    return render_template("cover.html")


@app.route("/configure", defaults={"entry_mode": "builtin"}, methods=["GET"])
@app.route("/configure/<entry_mode>", methods=["GET"])
def upload_page(entry_mode: str):
    return render_upload_config(entry_mode=entry_mode, error=None)


@app.route("/experiments", methods=["POST"])
def create_experiment():
    run_id = f"exp_{timestamp_slug()}"
    entry_mode = request.form.get("entry_mode", "builtin")
    builtin_fast_datasets = gnn_service.list_builtin_fast_datasets()
    wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", "")
    try:
        dataset, dataset_display_name, data_root = prepare_uploaded_dataset(run_id, entry_mode)
    except ValueError as exc:
        if wants_json:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return render_upload_config(entry_mode=entry_mode, error=str(exc))

    label_type = request.form.get("label_type", "multi")
    selected_strategy = request.form.get("strategy", "v4")
    if label_type == "single":
        selected_strategy = "v1"

    form_data = {
        "dataset": dataset,
        "dataset_display_name": dataset_display_name,
        "label_type": label_type,
        "strategy": selected_strategy,
        "target_ratio": float(request.form.get("target_ratio", "50")),
        "jaccard_threshold": float(request.form.get("jaccard_threshold", "0.5")),
        "undirected": request.form.get("undirected", "auto"),
        "calculate_spectral_errors": True,
        "purity_threshold": float(request.form.get("purity_threshold", "0.7")),
        "enable_similarity_weighted_features": request.form.get("enable_similarity_weighted_features", "false") == "true",
        "similarity_temperature": float(request.form.get("similarity_temperature", "5.0")),
        "model_type": "gcn",
        "train_epochs": 1,
        "calibrate_iters": int(request.form.get("calibrate_iters", "6")),
    }
    builtin_config = builtin_fast_datasets.get(dataset.lower())
    if builtin_config and int(form_data["target_ratio"]) not in set(builtin_config["allowed_ratios"]):
        allowed = "、".join(str(value) for value in builtin_config["allowed_ratios"])
        if wants_json:
            return jsonify({"ok": False, "error": f"{dataset} 仅支持压缩率 {allowed}。"}), 400
        return render_upload_config(entry_mode=entry_mode, error=f"{dataset_display_name} 仅支持压缩率 {allowed}。")
    save_state(run_id, create_initial_state(run_id, form_data, dataset, dataset_display_name, str(data_root), entry_mode))
    if wants_json:
        start_experiment_in_background(run_id)
        return jsonify(
            {
                "ok": True,
                "run_id": run_id,
                "running_url": url_for("running_page", run_id=run_id),
                "result_url": url_for("result_page", run_id=run_id),
                "state_url": url_for("experiment_state", run_id=run_id),
            }
        )
    return redirect(url_for("running_page", run_id=run_id))


@app.route("/experiments/<run_id>/state", methods=["GET"])
def experiment_state(run_id: str):
    state = load_state(run_id)
    if not state:
        abort(404)
    return jsonify(state)


@app.route("/experiments/<run_id>/gnn/state", methods=["GET"])
def gnn_test_state(run_id: str):
    state = load_gnn_state(run_id)
    if not state:
        return jsonify({"status": "idle"})
    return jsonify(state)


@app.route("/experiments/<run_id>/mapping/node/<supernode_id>", methods=["GET"])
def experiment_mapping_node(run_id: str, supernode_id: str):
    payload = load_mapping_payload(run_id)
    subgraph = (payload.get("subgraphs") or {}).get(supernode_id)
    if not subgraph:
        abort(404)
    return jsonify(subgraph)


@app.route("/experiments/<run_id>/mapping/edge/<edge_id>", methods=["GET"])
def experiment_mapping_edge(run_id: str, edge_id: str):
    payload = load_mapping_payload(run_id)
    subgraph = (payload.get("edge_subgraphs") or {}).get(edge_id)
    if not subgraph:
        abort(404)
    return jsonify(subgraph)


@app.route("/experiments/<run_id>/running", methods=["GET"])
def running_page(run_id: str):
    state = load_state(run_id)
    if not state:
        return redirect(url_for("upload_page"))

    if state.get("status") == "pending" and not state.get("execution_started"):
        execute_experiment(run_id)
        state = load_state(run_id)

    summary = read_json(summary_path(run_id), default=None)
    return render_template(
        "running.html",
        run_id=run_id,
        state=state,
        dataset_display_name=state.get("dataset_display_name", state.get("dataset")),
        stages=make_stage_cards(state),
        summary=summary,
    )


@app.route("/experiments/<run_id>/result", methods=["GET"])
def result_page(run_id: str):
    summary = read_json(summary_path(run_id), default=None)
    if not summary:
        return redirect(url_for("running_page", run_id=run_id))
    state = load_state(run_id)
    graph_preview = build_graph_preview(run_id)
    spectral_metrics = read_json(experiment_dir(run_id) / "spectral_metrics.json", default=None)
    return render_template(
        "result.html",
        result=summary,
        state=state,
        dataset_display_name=state.get("dataset_display_name", summary.get("dataset_display_name", summary["dataset"])),
        graph_preview=graph_preview,
        spectral_metrics=spectral_metrics,
        mapping_note="点击 coarse supernode 或 coarse edge 后，将按需加载对应的原图局部映射。",
        mapping_node_url_template=url_for("experiment_mapping_node", run_id=run_id, supernode_id="__SUPERNODE_ID__"),
        mapping_edge_url_template=url_for("experiment_mapping_edge", run_id=run_id, edge_id="__EDGE_ID__"),
        gnn_tests=list_gnn_tests(run_id),
    )


@app.route("/experiments/<run_id>/download/<export_kind>", methods=["GET"])
def download_export(run_id: str, export_kind: str):
    if export_kind not in {"pt", "text"}:
        abort(404)
    directory = export_dir(run_id, export_kind)
    if not directory.exists():
        abort(404)
    archive_base = experiment_dir(run_id) / f"{run_id}_{export_kind}"
    archive_path = shutil.make_archive(str(archive_base), "zip", str(directory))
    return send_file(archive_path, as_attachment=True, download_name=f"{run_id}_{export_kind}.zip")


@app.route("/experiments/<run_id>/gnn", methods=["GET", "POST"])
def gnn_test_page(run_id: str):
    summary = read_json(summary_path(run_id), default=None)
    if not summary:
        return redirect(url_for("running_page", run_id=run_id))

    if request.method == "POST":
        model_type = request.form.get("model_type", "gcn")
        epochs = int(request.form.get("epochs", "10"))
        wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", "")
        save_gnn_state(
            run_id,
            {
                "status": "queued",
                "model_type": model_type,
                "epochs": epochs,
                "current_epoch": 0,
                "latest_line": "任务已创建，等待启动...",
                "error": None,
            },
        )
        start_gnn_test_in_background(run_id, model_type, epochs)
        if wants_json:
            return jsonify(
                {
                    "ok": True,
                    "state_url": url_for("gnn_test_state", run_id=run_id),
                    "refresh_url": url_for("gnn_test_page", run_id=run_id),
                }
            )
        return redirect(url_for("gnn_test_page", run_id=run_id))

    tests = list_gnn_tests(run_id)
    latest_test = tests[0] if tests else None
    return render_template(
        "gnn_test.html",
        run_id=run_id,
        result=summary,
        dataset_display_name=summary.get("dataset_display_name", summary["dataset"]),
        models=gnn_service.list_supported_models(),
        tests=tests,
        latest_test=latest_test,
        gnn_state=load_gnn_state(run_id),
    )


if __name__ == "__main__":
    app.run(debug=True)
