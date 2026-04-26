# 数据清洗，删除所有无标签节点

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["facebook", "twitter", "tweibo"]


def load_labeled_nodes(label_path: Path) -> tuple[list[int], dict[int, str]]:
    labeled_nodes: list[int] = []
    label_lines: dict[int, str] = {}

    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            node_id = int(parts[0])
            if node_id in label_lines:
                continue
            labeled_nodes.append(node_id)
            label_lines[node_id] = stripped

    labeled_nodes.sort()
    return labeled_nodes, label_lines


def load_attrs_npz(path: Path) -> dict[str, np.ndarray | tuple[int, int] | str]:
    obj = np.load(str(path), allow_pickle=False)
    keys = set(obj.files)
    if {"data", "indices", "indptr", "shape"}.issubset(keys):
        out: dict[str, np.ndarray | tuple[int, int] | str] = {
            "format": "csr",
            "data": obj["data"],
            "indices": obj["indices"],
            "indptr": obj["indptr"],
            "shape": tuple(int(x) for x in obj["shape"].tolist()),
        }
        if "format" in keys:
            fmt = np.asarray(obj["format"])
            out["format_name"] = str(fmt.item()) if fmt.ndim == 0 else "csr"
        return out
    if "arr_0" in keys:
        return {"format": "dense", "arr_0": obj["arr_0"]}
    raise ValueError(f"unsupported attrs.npz format: keys={obj.files}")


def slice_attrs(attrs: dict[str, np.ndarray | tuple[int, int] | str], row_ids: list[int]) -> dict[str, np.ndarray]:
    if attrs["format"] == "dense":
        arr = np.asarray(attrs["arr_0"])
        return {"arr_0": arr[row_ids]}

    data = np.asarray(attrs["data"])
    indices = np.asarray(attrs["indices"])
    indptr = np.asarray(attrs["indptr"])
    shape = attrs["shape"]
    assert isinstance(shape, tuple)
    num_cols = int(shape[1])

    row_ids_arr = np.asarray(row_ids, dtype=np.int64)
    starts = indptr[row_ids_arr]
    ends = indptr[row_ids_arr + 1]
    counts = ends - starts

    out_indptr = np.zeros(len(row_ids) + 1, dtype=indptr.dtype)
    if counts.size:
        out_indptr[1:] = np.cumsum(counts, dtype=indptr.dtype)

    total_nnz = int(out_indptr[-1])
    out_indices = np.empty(total_nnz, dtype=indices.dtype)
    out_data = np.empty(total_nnz, dtype=data.dtype)

    cursor = 0
    for start, end in zip(starts.tolist(), ends.tolist()):
        width = int(end - start)
        if width <= 0:
            continue
        out_indices[cursor:cursor + width] = indices[start:end]
        out_data[cursor:cursor + width] = data[start:end]
        cursor += width

    return {
        "data": out_data,
        "indices": out_indices,
        "indptr": out_indptr,
        "shape": np.asarray([len(row_ids), num_cols], dtype=np.int64),
        "format": np.asarray(attrs.get("format_name", "csr")),
    }


def save_attrs_npz(path: Path, attrs: dict[str, np.ndarray]) -> None:
    np.savez(path, **attrs)


def wash_one(root: Path, dataset: str, out_suffix: str, out_name: str | None = None) -> None:
    raw_dir = root / dataset / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(str(raw_dir))

    target_name = out_name.strip() if out_name else f"{dataset}{out_suffix}"
    out_raw_dir = root / target_name / "raw"
    out_raw_dir.mkdir(parents=True, exist_ok=True)

    label_path = raw_dir / "labels.txt"
    edge_path = raw_dir / "edgelist.txt"
    attrs_path = raw_dir / "attrs.npz"

    labeled_nodes, label_lines = load_labeled_nodes(label_path)
    kept_node_set = set(labeled_nodes)
    node_map = {old_id: new_id for new_id, old_id in enumerate(labeled_nodes)}

    print(f"\n--- Washing {dataset} ---")
    print(f"Labeled nodes kept: {len(labeled_nodes)}")

    df = pd.read_csv(edge_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
    edge_mask = df["u"].isin(kept_node_set) & df["v"].isin(kept_node_set)
    washed_df = df.loc[edge_mask, ["u", "v"]].copy()
    before_dedup = len(washed_df)
    # Remove only exact directed duplicates like repeated (u, v),
    # but keep paired reverse edges (u, v) and (v, u) as two edges.
    washed_df = washed_df.drop_duplicates(subset=["u", "v"], keep="first")
    after_dedup = len(washed_df)
    washed_edges = washed_df.to_numpy(copy=True)
    if washed_edges.size:
        washed_edges[:, 0] = np.vectorize(node_map.get, otypes=[np.int64])(washed_edges[:, 0])
        washed_edges[:, 1] = np.vectorize(node_map.get, otypes=[np.int64])(washed_edges[:, 1])
    else:
        washed_edges = np.empty((0, 2), dtype=np.int64)
    np.savetxt(out_raw_dir / "edgelist.txt", washed_edges, fmt="%d")
    print(f"Edges kept before dedup: {before_dedup}")
    print(f"Exact directed duplicates removed: {before_dedup - after_dedup}")
    print(f"Edges kept after dedup: {len(washed_edges)}")

    with (out_raw_dir / "labels.txt").open("w", encoding="utf-8") as f:
        for old_id in labeled_nodes:
            parts = label_lines[old_id].split()
            rest = " ".join(parts[1:])
            f.write(f"{node_map[old_id]} {rest}\n")
    print(f"Labels kept: {len(labeled_nodes)}")

    attrs = load_attrs_npz(attrs_path)
    washed_attrs = slice_attrs(attrs, labeled_nodes)
    save_attrs_npz(out_raw_dir / "attrs.npz", washed_attrs)
    if "arr_0" in washed_attrs:
        shape = washed_attrs["arr_0"].shape
    else:
        shape = tuple(int(x) for x in washed_attrs["shape"].tolist())
    print(f"Attrs kept: shape={shape}")
    print(f"Saved to: {out_raw_dir.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove unlabeled nodes from raw datasets.")
    parser.add_argument("--root", default="data")
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--out-suffix", default="_washed")
    parser.add_argument("--out-name", default=None, help="Custom output dataset name. Only valid when --dataset is a single dataset.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()

    targets = DATASETS if args.dataset == "all" else [args.dataset]
    if args.dataset == "all" and args.out_name:
        raise ValueError("--out-name can only be used when --dataset is a single dataset.")
    for dataset in targets:
        try:
            wash_one(root, dataset, args.out_suffix, args.out_name)
        except FileNotFoundError as e:
            print(f"\nskip dataset: {dataset}")
            print("missing file or directory:", e)


if __name__ == "__main__":
    main()
