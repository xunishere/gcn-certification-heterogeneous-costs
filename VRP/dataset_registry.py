"""Canonical registry for datasets used by the certification experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    family: str
    main_q: int
    sensitivity_q: tuple[int, ...]
    main_q0: int
    sensitivity_q0: tuple[int, ...]
    default_sample_size: int


DATASET_REGISTRY = {
    "Cora": DatasetSpec(
        name="Cora",
        aliases=("Cora", "cora"),
        family="Planetoid",
        main_q=14,  # floor(0.01 * 1433 binary features)
        sensitivity_q=(14,),
        main_q0=14,  # lambda=1 reference point
        sensitivity_q0=(3, 7, 14, 28),  # floor(lambda*q), lambda=.25,.5,1,2
        default_sample_size=140,
    ),
    "CiteSeer": DatasetSpec(
        name="CiteSeer",
        aliases=("CiteSeer", "Citeseer", "citeseer"),
        family="Planetoid",
        main_q=37,  # floor(0.01 * 3703 binary features)
        sensitivity_q=(37,),
        main_q0=37,  # lambda=1 reference point
        sensitivity_q0=(9, 18, 37, 74),  # floor(lambda*q), lambda=.25,.5,1,2
        default_sample_size=140,
    ),
    "Karate": DatasetSpec(
        name="Karate",
        aliases=("Karate", "karate"),
        family="KarateClub",
        main_q=1,
        sensitivity_q=(1,),
        main_q0=1,
        sensitivity_q0=(1,),
        default_sample_size=34,
    ),
}

_ALIAS_TO_CANONICAL = {
    alias.casefold(): canonical
    for canonical, specification in DATASET_REGISTRY.items()
    for alias in specification.aliases
}


def canonical_dataset_name(name):
    try:
        return _ALIAS_TO_CANONICAL[str(name).casefold()]
    except KeyError as error:
        supported = ", ".join(DATASET_REGISTRY)
        raise ValueError(f"unsupported dataset {name!r}; choose one of {supported}") from error


def dataset_spec(name):
    return DATASET_REGISTRY[canonical_dataset_name(name)]


def cli_dataset_name(value):
    """``argparse`` type that accepts aliases and returns a canonical name."""
    import argparse

    try:
        return canonical_dataset_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def load_dataset(name, cache_dir):
    """Load a registered PyG dataset using its canonical spelling."""
    specification = dataset_spec(name)
    cache_dir = Path(cache_dir)
    if specification.family == "KarateClub":
        from torch_geometric.datasets import KarateClub

        return KarateClub()
    if specification.family == "Planetoid":
        from torch_geometric.datasets import Planetoid

        return Planetoid(str(cache_dir / "Planetoid"), name=specification.name)
    raise RuntimeError(f"unimplemented dataset family: {specification.family}")


def main_q_by_dataset(dataset_names):
    return {canonical_dataset_name(name): dataset_spec(name).main_q
            for name in dataset_names}


def main_q0_by_dataset(dataset_names):
    return {canonical_dataset_name(name): dataset_spec(name).main_q0
            for name in dataset_names}


def q_values_for_dataset(dataset_name, explicit_q=None, sensitivity=False):
    specification = dataset_spec(dataset_name)
    if explicit_q is not None:
        return tuple(int(value) for value in explicit_q)
    return (specification.sensitivity_q if sensitivity
            else (specification.main_q,))


def q0_values_for_dataset(dataset_name, explicit_q0=None, sensitivity=False):
    """Return global ``Q0`` values, resolving dataset-specific defaults."""
    specification = dataset_spec(dataset_name)
    if explicit_q0 is not None:
        return tuple(int(value) for value in explicit_q0)
    return (specification.sensitivity_q0 if sensitivity
            else (specification.main_q0,))
