"""Dependency-light node importance for heterogeneous attack costs."""
from collections import deque

try:
    from .graph_semantics import canonical_undirected_edges
except ImportError:  # Direct-script compatibility for the lightweight driver.
    from graph_semantics import canonical_undirected_edges


def _adjacency(num_nodes, edges):
    adj = [set() for _ in range(num_nodes)]
    for u, v in canonical_undirected_edges(num_nodes, edges):
        adj[u].add(v)
    return adj


def _minmax(values):
    lo, hi = min(values, default=0.0), max(values, default=0.0)
    if hi == lo:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def degree_centrality(adj):
    n = len(adj)
    return [len(x) / max(1, n - 1) for x in adj]


def pagerank(adj, damping=0.85, tolerance=1e-12, max_iterations=200):
    n = len(adj)
    if not n:
        return []
    rank = [1.0 / n] * n
    for _ in range(max_iterations):
        dangling = sum(rank[v] for v in range(n) if not adj[v])
        new = [(1.0 - damping) / n + damping * dangling / n for _ in range(n)]
        for v in range(n):
            if adj[v]:
                share = damping * rank[v] / len(adj[v])
                for w in adj[v]:
                    new[w] += share
        if sum(abs(a - b) for a, b in zip(new, rank)) <= tolerance:
            rank = new
            break
        rank = new
    return rank


def betweenness_centrality(adj):
    """Brandes betweenness for an undirected, unweighted graph."""
    n = len(adj)
    score = [0.0] * n
    for source in range(n):
        stack, pred = [], [[] for _ in range(n)]
        sigma, dist = [0.0] * n, [-1] * n
        sigma[source], dist[source] = 1.0, 0
        queue = deque([source])
        while queue:
            v = queue.popleft(); stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    queue.append(w); dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]; pred[w].append(v)
        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != source:
                score[w] += delta[w]
    return [x / 2.0 for x in score]


def compute_node_importance(num_nodes, edges, weights=(0.4, 0.4, 0.2)):
    """Return min-max component-weighted degree/PageRank/betweenness importance."""
    if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be three non-negative values with positive sum")
    weights = [float(w) / sum(weights) for w in weights]
    adj = _adjacency(num_nodes, edges)
    degree = _minmax(degree_centrality(adj))
    rank = _minmax(pagerank(adj))
    between = _minmax(betweenness_centrality(adj))
    return [weights[0] * d + weights[1] * p + weights[2] * b
            for d, p, b in zip(degree, rank, between)]


def attack_costs(importance, alpha=1.0):
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return [1.0 + alpha * float(value) for value in importance]
