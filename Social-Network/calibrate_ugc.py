# 确定桶宽
# python f:\毕设\Social-Network\calibrate_ugc.py --dataset facebook --version v1 --target_ratio 50

import subprocess
import re
import argparse
import os
import sys
from pathlib import Path

# --- 数据集版本化配置 ---
DATASET_CONFIGS = {
    'facebook': {
        'feat_size': 1283,
        'versions': {
            'v1': {'num_classes': 151, 'alpha': 0.3554, 'subdir': 'ugc_v1'},
            'v2': {'num_classes': 1, 'alpha': 0.1180, 'subdir': 'ugc_v2'},
            'v3': {'num_classes': 73, 'alpha': 0.3308, 'subdir': 'ugc_v3'},
            'v4': {'num_classes': 72, 'alpha': 0.3041, 'subdir': 'ugc_v4'}
        }
    },
    'twitter_sampled': {
        'feat_size': 216839,
        'versions': {
            'v1': {'num_classes': 782, 'alpha': 0.8489, 'subdir': 'ugc_v1'},
            'v2': {'num_classes': 109, 'alpha': 0.3722, 'subdir': 'ugc_v2'},
            'v3': {'num_classes': 426, 'alpha': 0.8019, 'subdir': 'ugc_v3'},
            'v4': {'num_classes': 898, 'alpha': 0.9197, 'subdir': 'ugc_v4'},
        }
    },
    'tweibo_sampled': {
        'feat_size': 1657,
        'versions': {
            'v1': {'num_classes': 8, 'alpha': 0.7902, 'subdir': 'ugc_v1'},
            'v2': {'num_classes': 8, 'alpha': 0.7902, 'subdir': 'ugc_v2'},
            'v3': {'num_classes': 8, 'alpha': 0.7902, 'subdir': 'ugc_v3'},
            'v4': {'num_classes': 8, 'alpha': 0.7902, 'subdir': 'ugc_v4'},
        }
    }
}

def run_ugc_and_get_ratio(python_exe, ugc_path, args_list):
    """运行 UGC.py 并提取压缩率数值"""
    try:
        # 运行子进程
        result = subprocess.run(
            [python_exe, str(ugc_path)] + args_list,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        # 使用正则从输出中寻找 "Graph reduced by: XX percent"
        match = re.search(r"Graph reduced by:\s*([\d\.]+)\s*percent", result.stdout)
        if match:
            return float(match.group(1))
        else:
            print("\n[Error] Could not find ratio in output. Full stdout:")
            print(result.stdout)
            print(result.stderr)
            return None
    except Exception as e:
        print(f"\n[Exception] {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="UGC Bin Width Calibration Utility")
    parser.add_argument('--dataset', type=str, default='facebook', choices=['facebook', 'twitter_sampled', 'tweibo_sampled'])
    parser.add_argument('--version', type=str, default='v2', choices=['v1', 'v2', 'v3', 'v4'], help='Data strategy version (v1/v2/v3)')
    parser.add_argument('--target_ratio', type=float, default=50.0, help='Target coarsening ratio (e.g., 50.0)')
    parser.add_argument('--iters', type=int, default=10, help='Number of binary search iterations')
    parser.add_argument('--low', type=float, default=0.00001, help='Initial low bound for bin_width')
    parser.add_argument('--high', type=float, default=0.5, help='Initial high bound for bin_width')
    args = parser.parse_args()

    # 1. 路径推导
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    ugc_path = project_root / "UGC-Universal-Graph-Coarsening" / "UGC.py"
    python_exe = sys.executable # 使用当前环境的 Python

    config_main = DATASET_CONFIGS.get(args.dataset)
    version_config = config_main['versions'].get(args.version)
    data_dir = script_dir / "data" / args.dataset / version_config['subdir']

    if version_config["num_classes"] is None or version_config["alpha"] is None:
        raise ValueError(
            f"Please fill num_classes/alpha for dataset={args.dataset}, version={args.version} "
            f"in DATASET_CONFIGS before calibration."
        )
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    print(f"\n--- Calibrating UGC for {args.dataset} ({args.version}) ---")
    print(f"Target Ratio: {args.target_ratio}%")
    print(f"Using Data: {data_dir}")

    # 2. 准备基础参数
    base_args = [
        "--dataset", args.dataset, # 显式传入 dataset 保证 UGC.py 输出正确
        "--dataset_not_in_torch_geometric", "True",
        "--edge_index_path", str(data_dir / "edge_index.pt"),
        "--node_feat_path", str(data_dir / "node_feat.pt"),
        "--label_path", str(data_dir / "label.pt"),
        "--feature_size", str(config_main['feat_size']),
        "--num_classes", str(version_config['num_classes']),
        "--model_type", "gcn",
        "--ratio", str(int(args.target_ratio)), 
        "--add_adj_to_node_features", "True",
        "--alpha", str(version_config['alpha']),
        "--epochs", "1" # 极速模式
    ]

    # 3. 二分查找循环
    low = args.low
    high = args.high
    best_width = 0
    best_diff = 100.0

    print(f"{'Iteration':<10} | {'bin_width':<12} | {'Actual Ratio':<15} | {'Diff':<10}")
    print("-" * 55)

    for i in range(1, args.iters + 1):
        mid = (low + high) / 2
        test_args = base_args + ["--bin_width", str(mid)]
        
        actual_ratio = run_ugc_and_get_ratio(python_exe, ugc_path, test_args)
        
        if actual_ratio is None:
            break
            
        diff = abs(actual_ratio - args.target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_width = mid
            
        print(f"{i:<10} | {mid:<12.6f} | {actual_ratio:<14.2f}% | {diff:<10.2f}")

        # 二分调整：宽度越大，压缩越狠（比率越高）
        if actual_ratio < args.target_ratio:
            low = mid
        else:
            high = mid

    print("-" * 55)
    print(f"Calibration Complete!")
    print(f"Recommended bin_width: {best_width:.6f}")
    if actual_ratio is not None:
        print(f"Final Actual Ratio: {actual_ratio:.2f}%")
    print("\nCopy this to UGC_bin_widths.py:")
    # 建议 key 包含版本，例如 facebook_v1_dot
    print(f"'{args.dataset}_{args.version}_dot': {{'{int(args.target_ratio)}': {best_width:.6f}}}")

if __name__ == "__main__":
    main()
