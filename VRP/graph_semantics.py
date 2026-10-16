"""One canonical graph shared by cost, GCN, and certification code."""
from __future__ import annotations

import hashlib
import json


GRAPH_SEMANTICS_VERSION = "undirected-simple-no-input-self-loops-v1"


def canonical_undirected_edges(num_nodes, edges):
    """Return sorted directed entries for one undirected simple graph.

    Input self-loops are removed, duplicate entries are collapsed, and every
    non-loop edge is represented in both directions.  GCN normalization adds
    one self-loop per node only after this canonicalization.
    """
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    canonical = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("every edge must contain exactly two node ids")
        raw_left, raw_right = edge
        left, right = int(raw_left), int(raw_right)
        if left != raw_left or right != raw_right:
            raise ValueError("node ids must be integers")
        if not (0 <= left < num_nodes and 0 <= right < num_nodes):
            raise ValueError("edge node id is outside [0, num_nodes)")
        if left == right:
            continue
        canonical.add((left, right))
        canonical.add((right, left))
    return tuple(sorted(canonical))


def canonical_undirected_edge_index(edge_index, num_nodes, *, device=None):
    """Torch ``edge_index`` form of :func:`canonical_undirected_edges`."""
    import torch

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    canonical = canonical_undirected_edges(
        num_nodes, edge_index.detach().cpu().t().tolist()
    )
    target_device = edge_index.device if device is None else torch.device(device)
    if not canonical:
        return torch.empty((2, 0), dtype=torch.long, device=target_device)
    return torch.tensor(
        canonical, dtype=torch.long, device=target_device
    ).t().contiguous()


def canonical_graph_sha256(num_nodes, edges):
    """Stable identity of the exact graph consumed by every experiment path."""
    payload = {
        "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
        "num_nodes": int(num_nodes),
        "directed_edge_entries": canonical_undirected_edges(num_nodes, edges),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
