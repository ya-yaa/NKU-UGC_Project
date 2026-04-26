# 标签在相同位置重合即视为同类
import argparse
from pathlib import Path
import sys

def load_label_lists(path: Path):
    """
    加载标签文件，保留标签的顺序和位置。
    返回字典: {node_id: [label1, label2, ...]}
    """
    label_lists = {}
    if not path.exists():
        return label_lists
    
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                node_id = int(parts[0])
                # 将该行所有标签转为整数列表，保持顺序
                current_node_labels = [int(p) for p in parts[1:]]
                label_lists[node_id] = current_node_labels
            except ValueError:
                continue
    return label_lists

def calculate_alpha_v3(edge_path: Path, label_lists: dict):
    """
    计算 alpha 值 (V3 策略: 位置对齐匹配法)。
    判定标准: 如果在任何一个相同位置 i，有 label_u[i] == label_v[i]，则视为“同类”。
    注意：不同位置的标签重叠（如 label_u[0] == label_v[1]）不计入同类。
    alpha = (无任何位置匹配的边数) / (两端均有标签的总边数)
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
                if u in label_lists and v in label_lists:
                    total_labeled_edges += 1
                    
                    list_u = label_lists[u]
                    list_v = label_lists[v]
                    
                    # 检查是否存在任一位置 i 使得 list_u[i] == list_v[i]
                    is_same_class = False
                    # 只比较到较短列表的长度
                    for i in range(min(len(list_u), len(list_v))):
                        if list_u[i] == list_v[i]:
                            is_same_class = True
                            break
                    
                    if not is_same_class:
                        diff_label_edges += 1
            except ValueError:
                continue
                
    if total_labeled_edges == 0:
        return 0.0, 0, total_edges
    
    alpha = diff_label_edges / total_labeled_edges
    return alpha, total_labeled_edges, total_edges

def main():
    parser = argparse.ArgumentParser(description="Calculate alpha (heterophily factor) using the Position-wise Match strategy.")
    parser.add_argument("--root", default="data", help="Root directory of datasets")
    parser.add_argument("--dataset", default="all", choices=["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled", "all"], help="Dataset name or 'all'")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()

    datasets = ["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled"] if args.dataset == "all" else [args.dataset]

    print(f"{'Dataset':<12} | {'Alpha (V3)':<12} | {'Labeled Edges':<15} | {'Coverage':<10}")
    print("-" * 60)

    for ds in datasets:
        raw_dir = root / ds / "raw"
        label_path = raw_dir / "labels.txt"
        edge_path = raw_dir / "edgelist.txt"
        
        if not label_path.exists() or not edge_path.exists():
            print(f"{ds:<12} | {'Missing files':<12}")
            continue
            
        # 1. 加载标签列表
        label_lists = load_label_lists(label_path)
        
        # 2. 计算 alpha (V3)
        res = calculate_alpha_v3(edge_path, label_lists)
        if res:
            alpha, labeled_count, total_count = res
            coverage = (labeled_count / total_count * 100) if total_count > 0 else 0
            print(f"{ds:<12} | {alpha:<12.4f} | {labeled_count:<15} | {coverage:>6.2f}%")

if __name__ == "__main__":
    main()
