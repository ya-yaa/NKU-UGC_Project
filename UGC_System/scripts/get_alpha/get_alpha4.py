# Jaccard-threshold strategy
# python f:\毕设\Social-Network\get_alpha4.py --dataset facebook --jaccard-threshold 0.5

import argparse
from pathlib import Path


def load_label_sets(path: Path):
    label_sets = {}
    if not path.exists():
        return label_sets

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                node_id = int(parts[0])
                current_node_labels = set(int(p) for p in parts[1:])
                label_sets[node_id] = current_node_labels
            except ValueError:
                continue
    return label_sets


def jaccard_similarity(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def calculate_alpha_v4(edge_path: Path, label_sets: dict[int, set[int]], jaccard_threshold: float):
    total_labeled_edges = 0
    diff_label_edges = 0
    total_edges = 0

    if not edge_path.exists():
        print(f"Error: {edge_path} not found.")
        return None

    with edge_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            total_edges += 1
            try:
                u = int(parts[0])
                v = int(parts[1])
                if u in label_sets and v in label_sets:
                    total_labeled_edges += 1
                    sim = jaccard_similarity(label_sets[u], label_sets[v])
                    if sim < jaccard_threshold:
                        diff_label_edges += 1
            except ValueError:
                continue

    if total_labeled_edges == 0:
        return 0.0, 0, total_edges

    alpha = diff_label_edges / total_labeled_edges
    return alpha, total_labeled_edges, total_edges


def main():
    parser = argparse.ArgumentParser(
        description="Calculate alpha (heterophily factor) using the Jaccard-threshold strategy."
    )
    parser.add_argument("--root", default="data", help="Root directory of datasets")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled", "all"],
        help="Dataset name or 'all'",
    )
    parser.add_argument("--jaccard-threshold", type=float, default=0.5, help="Threshold for same-class decision")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()

    datasets = (
        ["facebook", "twitter", "tweibo", "facebook_sampled", "twitter_sampled", "tweibo_sampled"]
        if args.dataset == "all"
        else [args.dataset]
    )

    print(
        f"{'Dataset':<16} | {'Alpha (V4)':<12} | {'Labeled Edges':<15} | {'Coverage':<10} | {'Thr':<6}"
    )
    print("-" * 76)

    for ds in datasets:
        raw_dir = root / ds / "raw"
        label_path = raw_dir / "labels.txt"
        edge_path = raw_dir / "edgelist.txt"

        if not label_path.exists() or not edge_path.exists():
            print(f"{ds:<16} | {'Missing files':<12}")
            continue

        label_sets = load_label_sets(label_path)
        res = calculate_alpha_v4(edge_path, label_sets, args.jaccard_threshold)
        if res:
            alpha, labeled_count, total_count = res
            coverage = (labeled_count / total_count * 100) if total_count > 0 else 0
            print(
                f"{ds:<16} | {alpha:<12.4f} | {labeled_count:<15} | {coverage:>6.2f}% | {args.jaccard_threshold:<6.2f}"
            )


if __name__ == "__main__":
    main()
