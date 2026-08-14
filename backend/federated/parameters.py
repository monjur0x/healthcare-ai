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
) -> list[np.ndarray]:
    """
    Average client weight lists element-wise (FedAvg aggregation).

    Parameters
    ----------
    weights_list : Sequence[list[np.ndarray]]
        One weight list per client; every list must share the same
        structure.

    Returns
    -------
    list[np.ndarray]
        Element-wise mean of the client weight lists.

    Raises
    ------
    ValueError
        If no clients are provided or the weight lists are misaligned.
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

    return [
        np.mean(np.stack([weights[position] for weights in weights_list]), axis=0)
        for position in range(len(reference))
    ]


__all__ = ["average_weights"]
