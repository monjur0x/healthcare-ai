"""
Federated aggregation helpers (NumPy-native).

FedAvg parameters are plain lists of NumPy arrays so they serialize
cleanly through Flower without framework-specific encoders.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def average_weights(
    weights_list: Sequence[list[np.ndarray]],
    sample_counts: Sequence[float] | None = None,
) -> list[np.ndarray]:
    """
    Average client weight lists element-wise (FedAvg aggregation).

    Parameters
    ----------
    weights_list : Sequence[list[np.ndarray]]
        One weight list per client; every list must share the same
        structure.
    sample_counts : Sequence[float] | None
        Per-client sample counts for count-weighted FedAvg. When None
        (default), every client counts once (uniform mean), preserving
        the single-argument ``AggregateFn`` contract.

    Returns
    -------
    list[np.ndarray]
        Element-wise (count-weighted) mean of the client weight lists.

    Raises
    ------
    ValueError
        If no clients are provided, the weight lists are misaligned,
        the counts do not match the clients, or the counts are
        negative/NaN/summing to zero.
    """

    if not weights_list:
        raise ValueError("Cannot average an empty list of client weights.")

    reference = weights_list[0]
    for index, weights in enumerate(weights_list[1:], start=1):
        if len(weights) != len(reference):
            raise ValueError(
                f"Client {index} has {len(weights)} weight arrays; "
                f"expected {len(reference)}."
            )
        for position, (array, ref) in enumerate(zip(weights, reference, strict=True)):
            if array.shape != ref.shape:
                raise ValueError(
                    f"Client {index} array {position} has shape "
                    f"{array.shape}; expected {ref.shape}."
                )

    if sample_counts is None:
        return [
            np.mean(np.stack([weights[position] for weights in weights_list]), axis=0)
            for position in range(len(reference))
        ]

    counts = _validated_counts(len(weights_list), sample_counts)
    total = float(sum(counts))
    return [
        sum(
            (count / total) * np.asarray(weights[position], dtype=np.float64)
            for count, weights in zip(counts, weights_list, strict=True)
        )
        for position in range(len(reference))
    ]


def scale_updates(
    updates: Sequence[list[np.ndarray]],
    sample_counts: Sequence[float],
) -> list[list[np.ndarray]]:
    """
    Scale each client's update by its sample share (``n_i / total``).

    Use before masked aggregation: masks added to pre-scaled updates
    still cancel exactly on the server, so the masked result is the
    true count-weighted FedAvg mean rather than a uniform mean.

    Parameters
    ----------
    updates : Sequence[list[np.ndarray]]
        One weight list per client.
    sample_counts : Sequence[float]
        Per-client sample counts.

    Returns
    -------
    list[list[np.ndarray]]
        Updates scaled by sample share, in client order.

    Raises
    ------
    ValueError
        If the counts do not match the updates or are
        negative/NaN/summing to zero.
    """

    counts = _validated_counts(len(list(updates)), sample_counts)
    updates = list(updates)
    total = float(sum(counts))
    return [
        [np.asarray(array, dtype=np.float64) * (count / total) for array in update]
        for update, count in zip(updates, counts, strict=True)
    ]


def _validated_counts(n_clients: int, sample_counts: Sequence[float]) -> list[float]:
    """Validate per-client counts, returning them as floats."""
    counts = [float(count) for count in sample_counts]
    if len(counts) != n_clients:
        raise ValueError(f"Got {len(counts)} sample counts for {n_clients} clients.")
    if any(not (count >= 0) for count in counts):
        raise ValueError("Sample counts must be non-negative numbers.")
    if sum(counts) <= 0:
        raise ValueError("Sample counts must sum to a positive total.")
    return counts


__all__ = ["average_weights", "scale_updates"]
