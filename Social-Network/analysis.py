# python f:\毕设\Social-Network\analysis.py --dataset twitter_sampled

import argparse
from pathlib import Path
import numpy as np
import torch


SUPPORTED_DATASETS = ["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled"]

def analyze_pyg_graph(name, data):
    n = int(data.num_nodes)
    e = int(data.num_edges)
    f = int(data.num_features) if getattr(data, "num_features", None) is not None else (int(data.x.size(1)) if getattr(data, "x", None) is not None else 0)
    deg = torch.bincount(data.edge_index[0], minlength=n)
    avg_deg = float(deg.float().mean()) if n > 0 else 0.0
    max_deg = int(deg.max()) if deg.numel() > 0 else 0
    sparsity = None
    if getattr(data, "x", None) is not None:
        x = data.x
        if isinstance(x, torch.Tensor) and x.is_sparse:
            nnz = x._nnz()
            total = x.size(0) * x.size(1)
            sparsity = 1 - nnz / total
        elif isinstance(x, torch.Tensor):
            sparsity = float((x == 0).sum().item()) / x.numel()
    lbl_shape = list(data.y.shape) if getattr(data, "y", None) is not None else [n]
    print("===== ", name, " =====")
    print("nodes:", n)
    print("edges:", e)
    print("features:", f)
    print("avg degree:", avg_deg)
    print("max degree:", max_deg)
    print("feature sparsity:", sparsity)
    print("label shape:", lbl_shape)
    print()

def load_attrs_npz(path: Path):
    obj = np.load(str(path), allow_pickle=False)
    keys = set(obj.files)
    if {"data", "indices", "indptr", "shape"}.issubset(keys):
        shape = tuple(int(x) for x in obj["shape"].tolist())
        nnz = int(obj["data"].shape[0])
        sparsity = 1.0 - (nnz / float(shape[0] * shape[1]))
        return {"num_nodes": shape[0], "num_features": shape[1], "sparsity": sparsity}
    if "arr_0" in keys:
        arr = obj["arr_0"]
        n, f = int(arr.shape[0]), int(arr.shape[1])
        zeros = int((arr == 0).sum())
        sparsity = float(zeros) / float(arr.size)
        return {"num_nodes": n, "num_features": f, "sparsity": sparsity}
    raise ValueError("unsupported attrs.npz format")


def stream_degree_stats(edge_file: Path, num_nodes: int):
    deg = np.zeros(num_nodes, dtype=np.int64)
    edges = 0
    with edge_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            if 0 <= u < num_nodes:
                deg[u] += 1
            edges += 1
    avg_degree = float(edges) / float(num_nodes) if num_nodes > 0 else 0.0
    max_degree = int(deg.max()) if deg.size > 0 else 0
    return edges, avg_degree, max_degree


def label_shape(path: Path, num_nodes: int):
    if not path.exists():
        return [num_nodes, 0]
    label_set = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            for x in parts[1:]:
                try:
                    label_set.add(int(x))
                except Exception:
                    continue
    return [num_nodes, len(label_set)]


def analyze_dataset(root: Path, name: str):
    raw = root / name / "raw"
    if not raw.exists():
        raise FileNotFoundError(str(raw))
    attrs = load_attrs_npz(raw / "attrs.npz")
    edges, avg_deg, max_deg = stream_degree_stats(raw / "edgelist.txt", attrs["num_nodes"])
    lbl_shape = label_shape(raw / "labels.txt", attrs["num_nodes"])
    print("===== ", name, " =====")
    print("nodes:", attrs["num_nodes"])
    print("edges:", edges)
    print("features:", attrs["num_features"])
    print("avg degree:", avg_deg)
    print("max degree:", max_deg)
    print("feature sparsity:", attrs["sparsity"])
    print("label shape:", lbl_shape)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--dataset", default="all", choices=SUPPORTED_DATASETS + ["all"])
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()

    targets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    for name in targets:
        try:
            analyze_dataset(root, name)
        except FileNotFoundError as e:
            print("===== ", name, " =====")
            print("load failed:", e)
            print()
    if args.dataset == "all":
        try:
            from torch_geometric.datasets import Planetoid, Reddit as PyGReddit
            cora = Planetoid(root=str(root), name="Cora")[0]
            analyze_pyg_graph("Cora", cora)
            citeseer = Planetoid(root=str(root), name="Citeseer")[0]
            analyze_pyg_graph("Citeseer", citeseer)
            reddit = PyGReddit(root=str(root / "Reddit"))[0]
            analyze_pyg_graph("Reddit", reddit)
        except Exception as e:
            print("===== PyG datasets =====")
            print("load failed:", e)
            print()


if __name__ == "__main__":
    main()
