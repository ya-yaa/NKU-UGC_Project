# 标签集合有交集即视为同类
# python f:\毕设\Social-Network\get_alpha2.py --dataset facebook

import argparse
from pathlib import Path
import sys

def load_label_sets(path: Path):
    """
    加载标签文件，将每个节点的所有标签存入一个集合。
    返回字典: {node_id: set([label1, label2, ...])}
    """
    label_sets = {}
    if not path.exists():
        return label_sets
    
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                node_id = int(parts[0])
                # 将该行所有标签转为整数集合
                current_node_labels = set(int(p) for p in parts[1:])
                label_sets[node_id] = current_node_labels
            except ValueError:
                continue
    return label_sets

def calculate_alpha_v2(edge_path: Path, label_sets: dict):
    """
    计算 alpha 值 (V2 策略: 集合交集法)。
    判定标准: 如果两个节点的标签集合交集为空，则视为“不同类”。
    alpha = (交集为空的边数) / (两端均有标签的总边数)
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
                if u in label_sets and v in label_sets:
                    total_labeled_edges += 1
                    
                    # 使用 isdisjoint 检查交集是否为空
                    # 如果 isdisjoint 为 True，说明交集为空 -> 不同类 -> 计入异质边
                    if label_sets[u].isdisjoint(label_sets[v]):
                        diff_label_edges += 1
            except ValueError:
                continue
                
    if total_labeled_edges == 0:
        return 0.0, 0, total_edges
    
    alpha = diff_label_edges / total_labeled_edges
    return alpha, total_labeled_edges, total_edges

def main():
    parser = argparse.ArgumentParser(description="Calculate alpha (heterophily factor) using the Label-Set-Intersection strategy.")
    parser.add_argument("--root", default="data", help="Root directory of datasets")
    parser.add_argument("--dataset", default="all", choices=["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled", "all"], help="Dataset name or 'all'")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()

    datasets = ["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled"] if args.dataset == "all" else [args.dataset]

    print(f"{'Dataset':<12} | {'Alpha (V2)':<12} | {'Labeled Edges':<15} | {'Coverage':<10}")
    print("-" * 60)

    for ds in datasets:
        raw_dir = root / ds / "raw"
        label_path = raw_dir / "labels.txt"
        edge_path = raw_dir / "edgelist.txt"
        
        if not label_path.exists() or not edge_path.exists():
            print(f"{ds:<12} | {'Missing files':<12}")
            continue
            
        # 1. 加载标签集合
        label_sets = load_label_sets(label_path)
        
        # 2. 计算 alpha (V2)
        res = calculate_alpha_v2(edge_path, label_sets)
        if res:
            alpha, labeled_count, total_count = res
            coverage = (labeled_count / total_count * 100) if total_count > 0 else 0
            print(f"{ds:<12} | {alpha:<12.4f} | {labeled_count:<15} | {coverage:>6.2f}%")

if __name__ == "__main__":
    main()
