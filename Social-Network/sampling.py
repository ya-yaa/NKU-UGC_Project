# Raw subgraph sampling (random walk).
# Reads edgelist.txt, labels.txt and attrs.npz from raw/,
# then writes a relabeled sampled subgraph back to a new raw/ directory.
# python f:\毕设\Social-Network\sampling.py --dataset twitter --target_nodes 3000

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_adj(edges: np.ndarray, num_nodes: int) -> list[list[int]]:
    adj = [[] for _ in range(num_nodes)]
    for u, v in tqdm(edges, desc="Building adjacency list"):
        adj[int(u)].append(int(v))
    return adj


def random_walk_sampling_fast(adj: list[list[int]], target_nodes: int) -> list[int]:
    sampled_nodes: set[int] = set()
    num_nodes = len(adj)
    target_nodes = min(target_nodes, num_nodes)

    valid_start_nodes = [i for i, neighbors in enumerate(adj) if neighbors]
    if not valid_start_nodes:
        return list(range(target_nodes))

    pbar = tqdm(total=target_nodes, desc="Sampling nodes")
    while len(sampled_nodes) < target_nodes:
        curr_node = random.choice(valid_start_nodes)
        if curr_node not in sampled_nodes:
            sampled_nodes.add(curr_node)
            pbar.update(1)

        walk_len = random.randint(50, 200)
        for _ in range(walk_len):
            neighbors = adj[curr_node]
            if not neighbors:
                break
            curr_node = random.choice(neighbors)
            if curr_node not in sampled_nodes:
                sampled_nodes.add(curr_node)
                pbar.update(1)
            if len(sampled_nodes) >= target_nodes:
                break
    pbar.close()
    return sorted(sampled_nodes)


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


def write_sampled_labels(src: Path, dst: Path, node_map: dict[int, int]) -> int:
    if not src.exists():
        return 0

    kept = 0
    with src.open("r", encoding="utf-8", errors="ignore") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            parts = line.split()
            if not parts:
                continue
            old_node = int(parts[0])
            new_node = node_map.get(old_node)
            if new_node is None:
                continue
            rest = " ".join(parts[1:])
            if rest:
                f_out.write(f"{new_node} {rest}\n")
            else:
                f_out.write(f"{new_node}\n")
            kept += 1
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--dataset", required=True, help="Input dataset directory name under data/")
    p.add_argument("--target_nodes", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-name", default=None, help="Custom output dataset name. Defaults to <dataset>_sampled")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    root = script_dir / args.root
    raw_dir = root / args.dataset / "raw"
    out_dataset_name = args.out_name.strip() if args.out_name else f"{args.dataset}_sampled"
    out_raw_dir = root / out_dataset_name / "raw"
    out_raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Raw Subgraph Sampling: {args.dataset} ---")

    edge_list_path = raw_dir / "edgelist.txt"
    print(f"Loading edgelist from: {edge_list_path}...")
    df = pd.read_csv(edge_list_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
    edges = df[["u", "v"]].to_numpy(copy=True)
    num_nodes = int(edges.max()) + 1 if edges.size else 0
    print(f"Original edges: {len(edges)}, max node ID: {num_nodes - 1}")

    adj = build_adj(edges, num_nodes)

    sampled_node_list = random_walk_sampling_fast(adj, args.target_nodes)
    sampled_nodes_set = set(sampled_node_list)
    node_map = {old_id: new_id for new_id, old_id in enumerate(sampled_node_list)}
    print(f"Sampled nodes: {len(sampled_node_list)}")

    print("Filtering and relabeling edges...")
    sampled_edges: list[tuple[int, int]] = []
    for u, v in tqdm(edges, desc="Filtering edges"):
        if int(u) in sampled_nodes_set and int(v) in sampled_nodes_set:
            sampled_edges.append((node_map[int(u)], node_map[int(v)]))

    edge_array = np.asarray(sampled_edges, dtype=np.int64)
    if edge_array.size == 0:
        edge_array = np.empty((0, 2), dtype=np.int64)
    np.savetxt(out_raw_dir / "edgelist.txt", edge_array, fmt="%d")
    print(f"Saved sampled edgelist: {len(sampled_edges)} edges")

    labels_path = raw_dir / "labels.txt"
    kept_labels = write_sampled_labels(labels_path, out_raw_dir / "labels.txt", node_map)
    if labels_path.exists():
        print(f"Saved sampled labels.txt: {kept_labels} rows")

    print("Filtering attrs.npz...")
    attrs = load_attrs_npz(raw_dir / "attrs.npz")
    sampled_attrs = slice_attrs(attrs, sampled_node_list)
    save_attrs_npz(out_raw_dir / "attrs.npz", sampled_attrs)
    if "arr_0" in sampled_attrs:
        print(f"Saved sampled attrs.npz: shape={sampled_attrs['arr_0'].shape}")
    else:
        print(f"Saved sampled attrs.npz: shape={tuple(int(x) for x in sampled_attrs['shape'].tolist())}")

    print(f"\nSampling complete! Output: {out_raw_dir.parent}")


if __name__ == "__main__":
    main()
