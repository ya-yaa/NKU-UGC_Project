# 只保留首个标签
# python f:\毕设\Social-Network\get_alpha1.py --dataset facebook

import argparse
from pathlib import Path
import sys

def load_first_labels(path: Path):
    """
    加载标签文件，只保留每行的第一个标签。
    返回字典: {node_id: first_label}
    """
    labels = {}
    if not path.exists():
        return labels
    
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                node_id = int(parts[0])
                first_label = int(parts[1])
                labels[node_id] = first_label
            except ValueError:
                continue
    return labels

def calculate_alpha(edge_path: Path, label_dict: dict):
    """
    流式计算 alpha 值。
    只考虑连接两个有标签节点的边。
    alpha = (标签不同的边数) / (总的有标签节点连接的边数)
    """
    total_labeled_edges = 0
    diff_label_edges = 0
    total_edges = 0
    
    if not edge_path.exists():
        print(f"Error: {edge_path} not found.")
        return None

    with edge_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            total_edges += 1
            try:
                u = int(parts[0])
                v = int(parts[1])
                
                # 只有当两个节点都有标签时，才参与 alpha 计算
                if u in label_dict and v in label_dict:
                    total_labeled_edges += 1
                    if label_dict[u] != label_dict[v]:
                        diff_label_edges += 1
            except ValueError:
                continue
                
    if total_labeled_edges == 0:
        return 0.0, 0, total_edges
    
    alpha = diff_label_edges / total_labeled_edges
    return alpha, total_labeled_edges, total_edges

def main():
    parser = argparse.ArgumentParser(description="Calculate alpha (heterophily factor) using the first-label strategy.")
    parser.add_argument("--root", default="data", help="Root directory of datasets")
    parser.add_argument("--dataset", default="all", choices=["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled", "all"], help="Dataset name or 'all'")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        # 如果是相对路径，相对于脚本所在目录
        root = (Path(__file__).resolve().parent / root).resolve()

    datasets = ["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled"] if args.dataset == "all" else [args.dataset]

    print(f"{'Dataset':<12} | {'Alpha':<10} | {'Labeled Edges':<15} | {'Coverage':<10}")
    print("-" * 55)

    for ds in datasets:
        raw_dir = root / ds / "raw"
        label_path = raw_dir / "labels.txt"
        edge_path = raw_dir / "edgelist.txt"
        
        if not label_path.exists() or not edge_path.exists():
            print(f"{ds:<12} | {'Missing files':<10}")
            continue
            
        # 1. 加载标签
        label_dict = load_first_labels(label_path)
        
        # 2. 计算 alpha
        res = calculate_alpha(edge_path, label_dict)
        if res:
            alpha, labeled_count, total_count = res
            coverage = (labeled_count / total_count * 100) if total_count > 0 else 0
            print(f"{ds:<12} | {alpha:<10.4f} | {labeled_count:<15} | {coverage:>6.2f}%")

if __name__ == "__main__":
    main()
