from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from .utils import write_json


@dataclass
class CoarseningArtifacts:
    num_supernodes: int
    reduction_percent: float
    graph_path: str


class CoarseningService:
    def __init__(self) -> None:
        self.seed = 42
        self.number_of_projectors = 500
        self.hash_function = "dot"
        self.projectors_distribution = "uniform"
        self.out_of_sample = 0
        self.max_visual_edges = 1200
        self.max_neighbors_per_supernode = 12

    def _fix_seeds(self) -> None:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        import random

        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _hashed_values(self, data: Data, feature_size: int) -> torch.Tensor:
        if self.projectors_distribution == "normal":
            projectors = torch.FloatTensor(self.number_of_projectors, feature_size).normal_(0, 1)
        else:
            projectors = torch.FloatTensor(self.number_of_projectors, feature_size).uniform_(0, 1)

        if self.out_of_sample != 0:
            num_out_of_sample = int(data.num_nodes * (1 - self.out_of_sample))
            idx = np.random.randint(data.num_nodes, size=num_out_of_sample)
            out_of_sampled_data_x = data.x[idx, :]
        else:
            out_of_sampled_data_x = data.x

        if self.hash_function == "L2-norm":
            return torch.cdist(out_of_sampled_data_x, projectors, p=2)
        if self.hash_function == "L1-norm":
            return torch.cdist(out_of_sampled_data_x, projectors, p=1)
        return torch.matmul(out_of_sampled_data_x, projectors.T)

    def _partition(self, bin_values: torch.Tensor, bin_width: float) -> dict[int, int]:
        import random

        bias = torch.tensor([random.uniform(-bin_width, bin_width) for _ in range(self.number_of_projectors)])
        temp = torch.floor((1 / bin_width) * (bin_values + bias))
        cluster, _ = torch.mode(temp, dim=1)
        return {i: int(cluster[i]) for i in range(bin_values.shape[0])}

    def generate(
        self,
        edge_index_path: Path,
        node_feat_path: Path,
        label_path: Path,
        alpha: float,
        bin_width: float,
        output_path: Path,
    ) -> CoarseningArtifacts:
        self._fix_seeds()

        node_feat = torch.load(node_feat_path)
        edge_index = torch.load(edge_index_path)
        labels = torch.load(label_path)
        labels_t = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()

        data = Data(x=node_feat, edge_index=edge_index, y=labels_t)
        base_feature_size = int(data.x.size(1))

        data.x = (1 - alpha) * data.x
        g_adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)[0]
        g_adj = alpha * g_adj
        data.x = torch.cat((data.x, g_adj), dim=1)
        feature_size = base_feature_size + data.num_nodes

        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[perm[:num_train]] = True
        data.val_mask[perm[num_train:num_train + num_val]] = True
        data.test_mask[perm[num_train + num_val:]] = True

        bin_values = self._hashed_values(data, feature_size)
        partition_map = self._partition(bin_values, bin_width)

        cluster_values = list(partition_map.values())
        unique_values = set(cluster_values)
        members_by_supernode: dict[int, list[int]] = {}
        supernode_sizes = torch.zeros(len(unique_values))

        for supernode_id, cluster_value in enumerate(unique_values):
            members = [node_id for node_id, value in partition_map.items() if value == cluster_value]
            members_by_supernode[supernode_id] = members
            supernode_sizes[supernode_id] = len(members)

        p_hat = torch.zeros((data.num_nodes, len(unique_values)))
        for supernode_id, members in members_by_supernode.items():
            for member in members:
                p_hat[member, supernode_id] = 1

        p_hat_sparse = p_hat.to_sparse()
        p = torch.sparse.mm(p_hat_sparse, torch.diag(torch.pow(supernode_sizes, -1 / 2)))
        features = data.x.to_sparse()
        coarse_features = torch.sparse.mm(torch.t(p), features.to_dense())

        edge_values = torch.ones(data.edge_index.shape[1])
        shape = torch.Size([data.x.shape[0], data.x.shape[0]])
        original_adj = torch.sparse_coo_tensor(data.edge_index, edge_values, shape)
        coarse_adj = torch.sparse.mm(torch.t(p_hat_sparse), torch.sparse.mm(original_adj, p_hat_sparse))
        diag_matrix = np.diag(np.array(supernode_sizes.cpu(), dtype=np.float32))
        coarse_dense = coarse_adj.to_dense().cpu().numpy() + diag_matrix - np.identity(diag_matrix.shape[0], dtype=np.float32)

        one_hot_labels = torch.eye(int(labels_t.max().item()) + 1)[labels_t, :]
        one_hot_labels[~data.train_mask] = 0
        coarse_labels = torch.argmax(torch.sparse.mm(torch.t(p).double(), one_hot_labels.double()).double(), 1).cpu().tolist()

        nodes = []
        details: dict[str, dict] = {}
        for supernode_id, members in members_by_supernode.items():
            member_labels: dict[int, int] = {}
            for member in members:
                member_label = int(labels_t[member].item())
                member_labels[member_label] = member_labels.get(member_label, 0) + 1
            top_labels = sorted(member_labels.items(), key=lambda item: item[1], reverse=True)
            preview_members = members[:30]
            node_key = f"sn_{supernode_id}"
            nodes.append(
                {
                    "data": {
                        "id": node_key,
                        "label": f"S{supernode_id}",
                        "size": len(members),
                        "count": len(members),
                        "coarse_label": int(coarse_labels[supernode_id]),
                    }
                }
            )
            details[node_key] = {
                "title": f"Supernode {supernode_id}",
                "count": len(members),
                "coarse_label": int(coarse_labels[supernode_id]),
                "members": members,
                "member_preview": preview_members,
                "member_count": len(members),
                "top_labels": [{"label": int(label), "count": int(count)} for label, count in top_labels[:6]],
                "feature_norm": float(torch.norm(coarse_features[supernode_id]).item()),
            }

        all_edges = []
        rows, cols = np.nonzero(coarse_dense)
        for src, dst in zip(rows.tolist(), cols.tolist()):
            if src >= dst:
                continue
            weight = float(coarse_dense[src][dst])
            if weight <= 0:
                continue
            all_edges.append(
                {
                    "id": f"sn_{src}_sn_{dst}",
                    "source": f"sn_{src}",
                    "target": f"sn_{dst}",
                    "weight": round(weight, 4),
                }
            )

        all_edges.sort(key=lambda item: item["weight"], reverse=True)
        visual_edges = all_edges[: self.max_visual_edges]
        edges = []
        for edge_item in visual_edges:
            edge_payload = {
                "data": {
                    "id": edge_item["id"],
                    "source": edge_item["source"],
                    "target": edge_item["target"],
                    "weight": edge_item["weight"],
                }
            }
            edges.append(edge_payload)
        for edge_item in all_edges:
            source = edge_item["source"]
            target = edge_item["target"]
            details[source].setdefault("neighbors", []).append({"id": target, "weight": edge_item["weight"]})
            details[target].setdefault("neighbors", []).append({"id": source, "weight": edge_item["weight"]})
        for node_key in details:
            details[node_key]["neighbors"] = details[node_key].get("neighbors", [])[: self.max_neighbors_per_supernode]

        reduction_percent = (1 - len(unique_values) / len(cluster_values)) * 100 if cluster_values else 0.0
        payload = {
            "nodes": nodes,
            "edges": edges,
            "details": details,
            "meta": {
                "num_supernodes": len(unique_values),
                "num_original_nodes": len(cluster_values),
                "reduction_percent": reduction_percent,
                "bin_width": bin_width,
                "alpha": alpha,
                "total_coarse_edges": len(all_edges),
                "visualized_edges": len(edges),
                "visualization_note": "节点映射为真实 UGC supernode；为保证前端交互可用，页面仅展示高权重 coarse edges 子集。",
            },
        }
        write_json(output_path, payload)
        return CoarseningArtifacts(
            num_supernodes=len(unique_values),
            reduction_percent=reduction_percent,
            graph_path=str(output_path),
        )
