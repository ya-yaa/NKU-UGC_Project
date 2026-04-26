# 取交集
# python f:\毕设\Social-Network\data_process2.py --dataset facebook --out-subdir ugc_v2

import argparse
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

def _read_edges_from_txt(path: Path, dtype: np.dtype) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= 200:
        arr = np.fromfile(str(path), dtype=dtype, sep=" ")
        if arr.size % 2 != 0:
            raise ValueError(f"invalid edgelist: {path}")
        arr = arr.reshape(-1, 2)
        return torch.from_numpy(arr.T.copy()).to(torch.long)

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    carry: np.ndarray | None = None

    with path.open("rb") as f:
        buf = ""
        while True:
            b = f.read(64 * 1024 * 1024)
            if not b:
                break
            s = buf + b.decode("utf-8", errors="ignore")
            cut = s.rfind("\n")
            if cut == -1:
                buf = s
                continue
            chunk = s[:cut].replace("\t", " ")
            buf = s[cut + 1 :]

            arr = np.fromstring(chunk, dtype=dtype, sep=" ")
            if carry is not None:
                arr = np.concatenate([carry, arr])
                carry = None
            if arr.size % 2 != 0:
                carry = arr[-1:]
                arr = arr[:-1]
            if arr.size:
                arr2 = arr.reshape(-1, 2)
                src_parts.append(arr2[:, 0].copy())
                dst_parts.append(arr2[:, 1].copy())

        tail = buf.strip()
        if tail:
            arr = np.fromstring(tail.replace("\t", " "), dtype=dtype, sep=" ")
            if carry is not None:
                arr = np.concatenate([carry, arr])
                carry = None
            if arr.size % 2 != 0:
                raise ValueError(f"invalid edgelist: {path}")
            if arr.size:
                arr2 = arr.reshape(-1, 2)
                src_parts.append(arr2[:, 0].copy())
                dst_parts.append(arr2[:, 1].copy())

    if carry is not None:
        raise ValueError(f"invalid edgelist: {path}")

    src = np.concatenate(src_parts) if src_parts else np.empty((0,), dtype=dtype)
    dst = np.concatenate(dst_parts) if dst_parts else np.empty((0,), dtype=dtype)
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).to(torch.long)
    return edge_index

def _to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    rev = edge_index.flip(0)
    return torch.cat([edge_index, rev], dim=1)

def _load_attrs_npz(path: Path) -> tuple[torch.Tensor, int, int, bool]:
    obj = np.load(str(path), allow_pickle=False)
    keys = set(obj.files)
    if {"data", "indices", "indptr", "shape"}.issubset(keys):
        data = torch.from_numpy(obj["data"].astype(np.float32, copy=False))
        indices = torch.from_numpy(obj["indices"].astype(np.int64, copy=False))
        indptr = torch.from_numpy(obj["indptr"].astype(np.int64, copy=False))
        shape = tuple(int(x) for x in obj["shape"].tolist())
        x = torch.sparse_csr_tensor(indptr, indices, data, size=shape, dtype=torch.float32)
        return x, shape[0], shape[1], True
    if "arr_0" in keys:
        arr = obj["arr_0"].astype(np.float32, copy=False)
        x = torch.from_numpy(arr)
        return x, int(x.size(0)), int(x.size(1)), False
    raise ValueError(f"unsupported attrs.npz format: keys={obj.files}")

def _is_sparse(x: torch.Tensor) -> bool:
    return x.layout != torch.strided

def _densify_if_small(x: torch.Tensor, densify_limit: int) -> tuple[torch.Tensor, bool]:
    if not _is_sparse(x):
        return x, False
    rows, cols = x.size(0), x.size(1)
    if rows * cols <= densify_limit:
        return x.to_dense(), True
    return x, False

def _should_force_dense(dataset: str) -> bool:
    return dataset == "facebook" or dataset.endswith("_sampled")

def _densify_for_ugc(
    x: torch.Tensor,
    dataset: str,
    densify_limit: int,
    max_dense_gb: float,
) -> tuple[torch.Tensor, bool]:
    if not _is_sparse(x):
        return x, False
    if _should_force_dense(dataset):
        rows, cols = x.size(0), x.size(1)
        dense_bytes = rows * cols * x.element_size()
        max_bytes = int(max_dense_gb * (1024 ** 3))
        if dense_bytes > max_bytes:
            need_gb = dense_bytes / (1024 ** 3)
            raise RuntimeError(
                f"dense conversion for {dataset} would need about {need_gb:.2f} GB, "
                f"which exceeds --max-dense-gb={max_dense_gb}"
            )
        return x.to_dense(), True
    return _densify_if_small(x, densify_limit)

def _empty_csr(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.zeros(rows + 1, dtype=torch.int64),
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0,), dtype=dtype),
        size=(rows, cols),
        dtype=dtype,
    )

def _take_rows(x: torch.Tensor, row_ids: list[int]) -> torch.Tensor:
    if not _is_sparse(x):
        return x[row_ids]
    if x.layout != torch.sparse_csr:
        return x.to_dense()[row_ids]
    if not row_ids:
        return _empty_csr(0, x.size(1), x.dtype)

    crow = x.crow_indices()
    col = x.col_indices()
    values = x.values()
    row_ids_t = torch.tensor(row_ids, dtype=torch.int64)
    starts = crow[row_ids_t]
    ends = crow[row_ids_t + 1]
    counts = ends - starts

    out_crow = torch.zeros(len(row_ids) + 1, dtype=torch.int64)
    out_crow[1:] = torch.cumsum(counts, dim=0)

    col_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if end > start:
            col_parts.append(col[start:end])
            value_parts.append(values[start:end])

    if col_parts:
        out_col = torch.cat(col_parts, dim=0)
        out_values = torch.cat(value_parts, dim=0)
    else:
        out_col = torch.empty((0,), dtype=col.dtype)
        out_values = torch.empty((0,), dtype=values.dtype)

    return torch.sparse_csr_tensor(
        out_crow,
        out_col,
        out_values,
        size=(len(row_ids), x.size(1)),
        dtype=x.dtype,
    )

def _append_empty_rows(x: torch.Tensor, extra_rows: int) -> torch.Tensor:
    if extra_rows <= 0:
        return x
    if not _is_sparse(x):
        return torch.cat([x, torch.zeros((extra_rows, x.size(1)), dtype=x.dtype)], dim=0)
    if x.layout != torch.sparse_csr:
        x = x.to_dense()
        return torch.cat([x, torch.zeros((extra_rows, x.size(1)), dtype=x.dtype)], dim=0)

    crow = x.crow_indices()
    last = int(crow[-1].item())
    extra_crow = torch.full((extra_rows,), last, dtype=crow.dtype)
    out_crow = torch.cat([crow, extra_crow], dim=0)
    return torch.sparse_csr_tensor(
        out_crow,
        x.col_indices(),
        x.values(),
        size=(x.size(0) + extra_rows, x.size(1)),
        dtype=x.dtype,
    )

def _resolve_target_nodes(
    dataset: str,
    n_full: int,
    nodes_in_edges: list[int],
    nodes_in_labels: list[int],
) -> list[int]:
    if dataset.endswith("_sampled"):
        return list(range(n_full))
    return sorted(list(set(nodes_in_edges) | set(nodes_in_labels)))


# --- 核心改进：并查集算法实现 V2 标签策略 ---
def _load_labels_txt_v2(path: Path, num_nodes: int) -> tuple[np.ndarray, int, int]:
    """
    V2 策略：两节点只要有一个标签相同就算一类（连通分量）。
    """
    parent = list(range(num_nodes))
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i]) # 路径压缩
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 建立 标签 -> 第一个拥有该标签的节点 的映射
    label_to_node = {}
    labeled_nodes_set = set()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 2:
                continue
            u = parts[0]
            if u < 0 or u >= num_nodes:
                continue
            
            labeled_nodes_set.add(u)
            labels = parts[1:]
            
            for l in labels:
                if l in label_to_node:
                    union(u, label_to_node[l])
                else:
                    label_to_node[l] = u

    # 最终标签计算
    y = np.full(num_nodes, -1, dtype=np.int64)
    for i in labeled_nodes_set:
        y[i] = find(i)

    # 映射到 0 到 C-1 的范围内
    uniq = sorted(set(y.tolist()))
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y_mapped = np.array([mapping[v] for v in y], dtype=np.int64)
    
    return y_mapped, len(uniq), len(labeled_nodes_set)

def process_one(
    dataset: str,
    root: Path,
    out_subdir: str,
    force_undirected: bool | None,
    densify_limit: int,
    max_dense_gb: float,
    edge_dtype: str,
) -> None:
    dataset = dataset.lower()
    raw_dir = root / dataset / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(str(raw_dir))

    out_dir = root / dataset / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载边
    edge_index = _read_edges_from_txt(raw_dir / "edgelist.txt", np.dtype(edge_dtype))
    # 只有 Facebook 默认无向化
    if force_undirected is True or (force_undirected is None and dataset == "facebook"):
        edge_index = _to_undirected(edge_index)

    # 2. 加载特征 (全量)
    x_full, n_full, f_from_x, x_was_sparse = _load_attrs_npz(raw_dir / "attrs.npz")
    
    # 3. 确定子图涉及的所有节点 (实现精准裁剪)
    labels_path = raw_dir / "labels.txt"
    nodes_in_edges = edge_index.view(-1).unique().tolist() if edge_index.numel() > 0 else []
    
    nodes_in_labels = []
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    nodes_in_labels.append(int(parts[0]))
    
    # 合并所有出现的节点并排序，作为子图的节点集
    all_sampled_nodes = _resolve_target_nodes(dataset, n_full, nodes_in_edges, nodes_in_labels)
    num_nodes = len(all_sampled_nodes)
    
    if num_nodes == 0:
        print(f"Warning: No nodes found in {dataset} raw files.")
        return

    # 4. 节点重映射 (Relabeling)
    node_map = {old_id: i for i, old_id in enumerate(all_sampled_nodes)}
    if edge_index.numel() > 0:
        edge_index = torch.tensor([[node_map[int(u)], node_map[int(v)]] for u, v in edge_index.T.tolist()]).T
    
    # 计算 V2 标签 (并查集处理全量节点后切片)
    if labels_path.exists():
        max_orig_id = max(all_sampled_nodes)
        temp_y, num_classes, labeled = _load_labels_txt_v2(labels_path, max_orig_id + 1)
        y_np = temp_y[all_sampled_nodes]
    else:
        y_np = np.zeros(num_nodes, dtype=np.int64)
        num_classes = 1
        labeled = 0

    # 5. 提取特征 (只保留子图节点的行)
    valid_nodes = [n for n in all_sampled_nodes if n < n_full]
    invalid_count = num_nodes - len(valid_nodes)
    
    x = _take_rows(x_full, valid_nodes)
    if invalid_count > 0:
        x = _append_empty_rows(x, invalid_count)

    # 6. 稠密化处理
    x, densified = _densify_for_ugc(x, dataset, densify_limit, max_dense_gb)

    # 7. 保存结果
    edge_index_path = out_dir / "edge_index.pt"
    node_feat_path = out_dir / "node_feat.pt"
    label_path = out_dir / "label.pt"

    torch.save(edge_index, edge_index_path)
    torch.save(x, node_feat_path)
    torch.save(y_np, label_path)

    print(f"--- Processed {dataset} using V2 Strategy (Set Intersection) ---")
    print("num_nodes (subgraph):", int(num_nodes))
    print("num_edges:", int(edge_index.size(1)))
    print("feature_size:", int(f_from_x))
    print("x_densified:", bool(densified))
    print("num_classes:", int(num_classes))
    print("labeled_nodes:", int(labeled))
    print("----------------------------------------------------------------")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--dataset", default="facebook", choices=["facebook", "twitter", "tweibo", "twitter_sampled", "tweibo_sampled", "all"])
    p.add_argument("--out-subdir", default="ugc_v2") # 默认输出到 ugc_v2
    p.add_argument("--undirected", default=None, choices=["true", "false", "auto"])
    p.add_argument("--densify-limit", type=int, default=50_000_000)
    p.add_argument("--max-dense-gb", type=float, default=4.0)
    p.add_argument("--edge-dtype", default="int32", choices=["int32", "int64"])
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir / args.root
    
    if args.undirected == "true":
        force_undirected = True
    elif args.undirected == "false":
        force_undirected = False
    else:
        force_undirected = None

    targets = ["facebook", "twitter", "tweibo", "twitter_sampled", "tweibo_sampled"] if args.dataset == "all" else [args.dataset]
    for d in targets:
        try:
            process_one(
                dataset=d,
                root=root,
                out_subdir=args.out_subdir,
                force_undirected=force_undirected,
                densify_limit=int(args.densify_limit),
                max_dense_gb=float(args.max_dense_gb),
                edge_dtype=args.edge_dtype,
            )
        except FileNotFoundError as e:
            print("skip dataset:", d)
            print("missing file or directory:", e)

if __name__ == "__main__":
    main()
