"""Two-/three-layer sigmoid GCN used by the deep robustness verifier.

A depth-2 model is ``Ahat-X-W1-b1 -> sigmoid -> Ahat-H1-W2-b2``.
Depth 3 inserts one further sigmoid hidden layer.  Training, clean inference,
interval propagation, Monte Carlo evaluation, and LP construction all consume
the same normalized adjacency and parameter ordering defined here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops

from .graph_semantics import (
    GRAPH_SEMANTICS_VERSION,
    canonical_graph_sha256,
    canonical_undirected_edge_index,
)

NORMALIZATION_VERSION = "canonical-undirected-simple-self-loop-dad-v2"
DEEP_MODEL_CACHE_VERSION = "deep-sigmoid-gcn-cache-v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and every visible CUDA device."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(specification: str | torch.device) -> torch.device:
    if isinstance(specification, torch.device):
        device = specification
    elif specification == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``D^-1/2 (A+I) D^-1/2`` as a coalesced sparse tensor.

    The shared graph rule first removes input self-loops, collapses duplicate
    edges, and symmetrizes every edge.  Exactly one self-loop per node is then
    added for GCN normalization.  The result is therefore symmetric even when
    the raw input is directed or contains duplicates.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    device = edge_index.device if device is None else torch.device(device)
    edges = canonical_undirected_edge_index(
        edge_index, num_nodes, device=device
    )
    edges, _ = add_remaining_self_loops(edges, num_nodes=num_nodes)
    provisional = torch.sparse_coo_tensor(
        edges,
        torch.ones(edges.shape[1], dtype=dtype, device=device),
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
        check_invariants=False,
    ).coalesce()
    indices = provisional.indices()
    # Coalescing identifies duplicate unweighted edges; reset their value to 1.
    edge_values = torch.ones(indices.shape[1], dtype=dtype, device=device)
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, indices[0], edge_values)
    if bool((degree <= 0).any()):
        raise RuntimeError("self loops should make every normalized degree positive")
    inverse_sqrt = degree.rsqrt()
    values = inverse_sqrt[indices[0]] * edge_values * inverse_sqrt[indices[1]]
    return torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), device=device, dtype=dtype,
        check_invariants=False,
    ).coalesce()


def graph_aggregate(adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return (torch.sparse.mm(adjacency, features) if adjacency.is_sparse
            else adjacency @ features)


class DeepSigmoidGCN(nn.Module):
    """GCN with one (depth 2) or two (depth 3) sigmoid hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        depth: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if depth not in (2, 3):
            raise ValueError("depth must be 2 or 3")
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if len(hidden_dims) != depth - 1 or any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain depth-1 positive dimensions")
        if input_dim <= 0 or output_dim <= 1:
            raise ValueError("input_dim must be positive and output_dim at least two")
        self.input_dim = int(input_dim)
        self.hidden_dims = hidden_dims
        self.output_dim = int(output_dim)
        self.depth = int(depth)
        self.bias_enabled = bool(bias)
        dimensions = (self.input_dim,) + hidden_dims + (self.output_dim,)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(dimensions[index], dimensions[index + 1]))
            for index in range(depth)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(dimensions[index + 1]))
            for index in range(depth)
        ]) if self.bias_enabled else nn.ParameterList()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        for bias in self.biases:
            nn.init.zeros_(bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer, weight in enumerate(self.weights):
            hidden = graph_aggregate(adjacency, hidden) @ weight
            if self.bias_enabled:
                hidden = hidden + self.biases[layer]
            if layer < self.depth - 1:
                hidden = torch.sigmoid(hidden)
        return hidden

    def numpy_parameters(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weights = [parameter.detach().cpu().double().numpy().copy()
                   for parameter in self.weights]
        if self.bias_enabled:
            biases = [parameter.detach().cpu().double().numpy().copy()
                      for parameter in self.biases]
        else:
            dimensions = list(self.hidden_dims) + [self.output_dim]
            biases = [np.zeros(dimension, dtype=np.float64) for dimension in dimensions]
        return weights, biases

    def architecture(self) -> dict:
        return {
            "model": "DeepSigmoidGCN",
            "depth": self.depth,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
            "bias": self.bias_enabled,
        }


@dataclass(frozen=True)
class TrainingHyperparameters:
    epochs: int = 500
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 100

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid training hyperparameters")
        if self.patience <= 0:
            raise ValueError("patience must be positive")


@dataclass
class DeepModelRun:
    model: DeepSigmoidGCN
    x: torch.Tensor
    y: torch.Tensor
    adjacency: torch.Tensor
    logits: torch.Tensor
    probability: torch.Tensor
    prediction: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    clean_train_accuracy: float
    clean_validation_accuracy: float
    clean_test_accuracy: float
    cache_key: str
    cache_path: str
    cache_hit: bool
    train_seconds: float
    best_epoch: int
    cache_description: dict


def _tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _split_digest(train_mask: torch.Tensor, val_mask: torch.Tensor,
                  test_mask: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        digest.update(name.encode("ascii"))
        digest.update(mask.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_cache_description(
    *,
    dataset_name: str,
    seed: int,
    model: DeepSigmoidGCN,
    hyperparameters: TrainingHyperparameters,
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """Build a cache identity containing architecture/train/split/norm data."""
    return {
        "cache_version": DEEP_MODEL_CACHE_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
        "dataset": str(dataset_name),
        "seed": int(seed),
        "architecture": model.architecture(),
        "training": asdict(hyperparameters),
        "split_sha256": _split_digest(train_mask, val_mask, test_mask),
        "data": {
            "nodes": int(data.num_nodes),
            "features": int(data.num_features),
            "classes": int(data.y.max()) + 1,
            "canonical_graph_sha256": canonical_graph_sha256(
                data.num_nodes, data.edge_index.t().tolist()
            ),
            "features_sha256": _tensor_digest(data.x),
            "labels_sha256": _tensor_digest(data.y),
        },
        "training_device_type": device.type,
        "training_dtype": "torch.float32",
    }


def _cache_key(description: dict) -> str:
    encoded = json.dumps(description, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def train_or_load_deep(
    data,
    dataset_name: str,
    seed: int,
    device: str | torch.device,
    *,
    depth: int,
    hidden_dims: Sequence[int],
    bias: bool,
    hyperparameters: TrainingHyperparameters,
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model_cache_dir: str | Path,
) -> DeepModelRun:
    """Train or load a deterministic deep sigmoid GCN cache entry."""
    device = choose_device(device)
    seed_all(seed)
    x = data.x.float().to(device)
    y = data.y.long().to(device)
    if not bool(torch.all((x == 0) | (x == 1))):
        raise ValueError("binary feature flips require a binary clean feature matrix")
    train_mask, val_mask, test_mask = (mask.detach().cpu().bool().clone()
                                       for mask in masks)
    model = DeepSigmoidGCN(
        int(data.num_features), hidden_dims, int(data.y.max()) + 1,
        depth=depth, bias=bias,
    ).to(device)
    adjacency = normalized_adjacency(
        data.edge_index, int(data.num_nodes), device=device, dtype=x.dtype
    )
    description = model_cache_description(
        dataset_name=dataset_name,
        seed=seed,
        model=model,
        hyperparameters=hyperparameters,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        device=device,
    )
    cache_key = _cache_key(description)
    cache_directory = Path(model_cache_dir)
    cache_directory.mkdir(parents=True, exist_ok=True)
    hidden_slug = "x".join(str(value) for value in model.hidden_dims)
    cache_path = (cache_directory /
                  f"{dataset_name}-d{depth}-h{hidden_slug}-seed{seed}-{cache_key}.pt")
    cache_hit = False
    train_seconds = 0.0
    best_epoch = -1

    if cache_path.exists():
        payload = torch.load(cache_path, map_location=device, weights_only=True)
        if payload.get("description") != description:
            raise RuntimeError(
                f"refusing to overwrite incompatible model cache: {cache_path}"
            )
        model.load_state_dict(payload["state_dict"])
        train_seconds = float(payload.get("train_seconds", 0.0))
        best_epoch = int(payload.get("best_epoch", -1))
        cache_hit = True

    train_device = train_mask.to(device)
    val_device = val_mask.to(device)
    test_device = test_mask.to(device)
    if not bool(train_device.any()):
        raise ValueError("training mask is empty")

    if not cache_hit:
        started = time.perf_counter()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyperparameters.learning_rate,
            weight_decay=hyperparameters.weight_decay,
        )
        best_score = -math.inf
        best_state = None
        stale = 0
        for epoch in range(hyperparameters.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, adjacency)
            loss = F.cross_entropy(logits[train_device], y[train_device])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                evaluated = model(x, adjacency)
                if bool(val_device.any()):
                    score = float((evaluated[val_device].argmax(1) == y[val_device])
                                  .float().mean().item())
                else:
                    score = -float(F.cross_entropy(
                        evaluated[train_device], y[train_device]
                    ).item())
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= hyperparameters.patience:
                break
        if best_state is None:
            raise RuntimeError("training produced no model state")
        model.load_state_dict({name: value.to(device)
                               for name, value in best_state.items()})
        train_seconds = time.perf_counter() - started
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save({
            "description": description,
            "state_dict": best_state,
            "train_seconds": train_seconds,
            "best_epoch": best_epoch,
            "created_at": utc_now(),
        }, temporary)
        temporary.replace(cache_path)

    model.eval()
    with torch.no_grad():
        logits = model(x, adjacency)
        probability = logits.softmax(1)
        prediction = logits.argmax(1)
    train_accuracy = float((prediction[train_device] == y[train_device])
                           .float().mean().item())
    validation_accuracy = (float((prediction[val_device] == y[val_device])
                                 .float().mean().item())
                           if bool(val_device.any()) else float("nan"))
    test_accuracy = (float((prediction[test_device] == y[test_device])
                           .float().mean().item())
                     if bool(test_device.any()) else float("nan"))
    return DeepModelRun(
        model=model,
        x=x,
        y=y,
        adjacency=adjacency,
        logits=logits,
        probability=probability,
        prediction=prediction,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        clean_train_accuracy=train_accuracy,
        clean_validation_accuracy=validation_accuracy,
        clean_test_accuracy=test_accuracy,
        cache_key=cache_key,
        cache_path=str(cache_path),
        cache_hit=cache_hit,
        train_seconds=train_seconds,
        best_epoch=best_epoch,
        cache_description=description,
    )
