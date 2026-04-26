# 保留首个标签
# python f:\毕设\Social-Network\data_process1.py --dataset facebook --out-subdir ugc_v1

import argparse
from pathlib import Path

import numpy as np
import torch


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
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    directed_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for src, dst in zip(src_list, dst_list):
        pair = (int(src), int(dst))
        if pair in seen:
            continue
        seen.add(pair)
        directed_edges.append(pair)

    base_edges = list(directed_edges)
    for src, dst in base_edges:
        reverse_pair = (dst, src)
        if reverse_pair in seen:
            continue
        seen.add(reverse_pair)
        directed_edges.append(reverse_pair)

    return torch.tensor(directed_edges, dtype=edge_index.dtype, device=edge_index.device).t().contiguous()


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
    return True


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


def _load_labels_txt(path: Path, num_nodes: int) -> tuple[np.ndarray, int, int]:
    y = np.zeros(num_nodes, dtype=np.int64)
    labeled = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            node = int(parts[0])
            if node < 0 or node >= num_nodes:
                continue
            y[node] = int(parts[1])
            labeled += 1
    uniq = sorted(set(y.tolist()))
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y2 = np.array([mapping[int(v)] for v in y], dtype=np.int64)
    return y2, len(uniq), labeled


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
    # 创建 原始ID -> 新ID (0..N-1) 的映射
    node_map = {old_id: i for i, old_id in enumerate(all_sampled_nodes)}
    
    # 转换边索引
    if edge_index.numel() > 0:
        edge_index = torch.tensor([[node_map[int(u)], node_map[int(v)]] for u, v in edge_index.T.tolist()]).T
    
    # 转换标签并提取
    if labels_path.exists():
        # 先加载到一个临时的大数组中
        max_orig_id = max(all_sampled_nodes)
        temp_y, num_classes, labeled = _load_labels_txt(labels_path, max_orig_id + 1)
        # 只提取采样节点的标签，并按新顺序排列
        y_np = temp_y[all_sampled_nodes]
    else:
        y_np = np.zeros(num_nodes, dtype=np.int64)
        num_classes = 1
        labeled = 0

    # 5. 提取特征 (只保留子图节点的行)
    # 确保不越界 (部分采样节点可能超出了 attrs.npz 的范围，需填充)
    valid_nodes = [n for n in all_sampled_nodes if n < n_full]
    invalid_count = num_nodes - len(valid_nodes)
    
    x = _take_rows(x_full, valid_nodes)
    if invalid_count > 0:
        x = _append_empty_rows(x, invalid_count)
    if False and invalid_count > 0:
        # 填充缺失节点的特征
        if _is_sparse(x):
            empty = torch.sparse_csr_tensor(
                torch.zeros(invalid_count + 1, dtype=torch.int64),
                torch.empty((0,), dtype=torch.int64),
                torch.empty((0,), dtype=torch.float32),
                size=(invalid_count, f_from_x),
            )
            x = torch.cat([x, empty], dim=0)
        else:
            x = torch.cat([x, torch.zeros((invalid_count, f_from_x), dtype=x.dtype)], dim=0)

    # 6. 稠密化处理
    x, densified = _densify_for_ugc(x, dataset, densify_limit, max_dense_gb)

    # 7. 保存结果
    edge_index_path = out_dir / "edge_index.pt"
    node_feat_path = out_dir / "node_feat.pt"
    label_path = out_dir / "label.pt"

    torch.save(edge_index, edge_index_path)
    torch.save(x, node_feat_path)
    torch.save(y_np, label_path)

    print("dataset:", dataset)
    print("num_nodes (subgraph):", int(num_nodes))
    print("num_edges:", int(edge_index.size(1)))
    print("feature_size:", int(f_from_x))
    print("x_densified:", bool(densified))
    print("num_classes:", int(num_classes))
    print("labeled_nodes:", int(labeled))
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--dataset", default="facebook", choices=["facebook", "twitter", "tweibo", "twitter_sampled", "tweibo_sampled", "all"])
    p.add_argument("--out-subdir", default="ugc")
    p.add_argument("--undirected", default=None, choices=["true", "false", "auto"])
    p.add_argument("--densify-limit", type=int, default=50_000_000)
    p.add_argument("--max-dense-gb", type=float, default=4.0)
    p.add_argument("--edge-dtype", default="int32", choices=["int32", "int64"])
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()
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
            print()


if __name__ == "__main__":
    main()
