"""COMO functions and adjacency functions.

This file contains the various cost of missing out (COMO) functions and
adjacency functions used in the ABM simulations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core_abm_class import ABDRunner

COMOFunction = Callable[["ABDRunner", np.ndarray, int, int, Mapping[str, Any]], float]


def _softplus(value: np.ndarray | float, softness: float) -> np.ndarray:
    """Stable softplus used to smooth rectifiers.

    Parameters
    ----------
    value:
        Input array or scalar.
    softness:
        Positive parameter controlling the transition width.  Smaller values
        make the function approach ``max(0, value)``.
    """

    if softness <= 0.0:
        raise ValueError("softness must be positive")

    scaled = np.asarray(value, dtype=float) / softness
    # ``log1p(exp(-abs(x))) + max(x, 0)`` keeps good numerical precision when
    # ``value`` has large magnitude, mirroring scipy.special.softplus.
    return softness * (np.log1p(np.exp(-np.abs(scaled))) + np.maximum(scaled, 0.0))


def build_simple_adjacency(
    n_agents: int,
    *,
    weight: float = 1.0,
    self_weight: float = 0.0,
) -> np.ndarray:
    """Create a fully connected adjacency matrix with uniform weights.

    Parameters
    ----------
    n_agents:
        Number of agents in the population.
    weight:
        Weight assigned to edges between distinct agents.
    self_weight:
        Optional weight to place on the diagonal.  The default leaves it at
        zero so only neighbour interactions are represented.
    """
    adjacency = np.full((n_agents, n_agents), float(weight), dtype=float)
    np.fill_diagonal(adjacency, float(self_weight))
    return adjacency


def build_ring_adjacency(
    n_agents: int,
    neighbours: int,
    *,
    weight: float = 1.0,
    self_weight: float = 0.0,
) -> np.ndarray:
    """Create a k-nearest-neighbour ring lattice adjacency matrix.

    Parameters
    ----------
    n_agents:
        Number of agents in the population.
    neighbours:
        Number of neighbours to connect on each side of the ring.  The final
        degree is ``2 * neighbours`` and cannot exceed ``n_agents - 1``.
    weight:
        Weight assigned to each neighbour edge.
    self_weight:
        Optional weight to place on the diagonal.  Defaults to zero.
    """

    if neighbours >= n_agents:
        raise ValueError("neighbours must be smaller than n_agents")

    adjacency = np.zeros((n_agents, n_agents), dtype=float)

    indices = np.arange(n_agents)
    for offset in range(1, neighbours + 1):
        right_indices = (indices + offset) % n_agents
        left_indices = (indices - offset) % n_agents
        adjacency[indices, right_indices] = float(weight)
        adjacency[indices, left_indices] = float(weight)

    np.fill_diagonal(adjacency, float(self_weight))
    return adjacency


def build_grouped_adjacency(
    group_assignments: Sequence[str | None],
    *,
    intra_group_weight: float = 1.0,
    inter_group_weight: float = 0.1,
    mixing_probability: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create an adjacency matrix that emphasises within-group ties.

    Parameters
    ----------
    group_assignments:
        Sequence assigning each agent to a group.  Agents with the same label
        form dense clusters.  ``None`` values are treated as their own group,
        meaning ungrouped agents will not be linked to others unless
        ``inter_group_weight``/``mixing_probability`` permits it.
    intra_group_weight:
        Weight applied to edges connecting members of the same group.  Set to
        zero to disable intra-group connections.
    inter_group_weight:
        Baseline weight for connections across groups.  This is only applied when
        ``mixing_probability`` allows an edge to exist.
    mixing_probability:
        Probability that a cross-group edge is included.  ``0`` removes all
        cross-group links, ``1`` creates a fully connected inter-group graph, and
        intermediate values sample sparse bridges between groups.
    rng:
        Optional random generator used when sampling sparse inter-group
        connections.
    """

    if not 0.0 <= mixing_probability <= 1.0:
        raise ValueError("mixing_probability must lie in the interval [0, 1]")

    n_agents = len(group_assignments)
    adjacency = np.zeros((n_agents, n_agents), dtype=float)

    needs_rng = 0.0 < mixing_probability < 1.0
    rng = np.random.default_rng() if needs_rng and rng is None else rng

    group_to_indices: dict[Any, list[int]] = defaultdict(list)
    for index, label in enumerate(group_assignments):
        group_to_indices[label].append(index)

    if intra_group_weight != 0.0:
        intra_weight = float(intra_group_weight)
        for indices in group_to_indices.values():
            idx_array = np.asarray(indices, dtype=int)
            if idx_array.size <= 1:
                continue
            adjacency[np.ix_(idx_array, idx_array)] = intra_weight

    if inter_group_weight > 0.0 and mixing_probability > 0.0:
        same_group = np.zeros((n_agents, n_agents), dtype=bool)
        for indices in group_to_indices.values():
            idx_array = np.asarray(indices, dtype=int)
            same_group[np.ix_(idx_array, idx_array)] = True
        np.fill_diagonal(same_group, True)
        cross_mask = ~same_group
        if mixing_probability == 1.0:
            adjacency[cross_mask] = float(inter_group_weight)
        else:
            assert rng is not None  # for type checking; ensured by needs_rng
            random_draws = rng.random((n_agents, n_agents))
            selection = cross_mask & (random_draws < mixing_probability)
            adjacency[selection] = float(inter_group_weight)

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def expand_adjacency_with_hops(
    adjacency: np.ndarray,
    hop_weights: Mapping[int, float] | Sequence[float],
    *,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """Combine higher-order neighbour influence into a single adjacency matrix.
    Parameters
    ----------
    adjacency:
        Base adjacency matrix describing first-order connections.  The matrix
        need not be row-normalised -- :class:`ABDRunner` will take care of that
        when it is supplied to a simulation.
    hop_weights:
        Mapping from hop distance to a weight, or a sequence of weights ordered
        by hop distance.  ``1`` corresponds to immediate neighbours, ``2`` to
        second-order neighbours and so on.  ``0`` weights are ignored so you
        can quickly try different decay schedules.
    zero_diagonal:
        When ``True`` (the default) the diagonal is cleared after combining the
        weighted powers.  This prevents self-influence that can arise from
        multi-step walks that return to the source agent.  Set to ``False`` if
        you purposefully want to retain self loops.
    Returns
    -------
    numpy.ndarray
        Weighted combination of the requested adjacency powers.  Feed the
        matrix straight into :class:`ABDRunner` to let the COMO functions react
        to multi-hop neighbours.
    """

    base = np.array(adjacency, dtype=float, copy=True)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if isinstance(hop_weights, Mapping):
        weights_iter = hop_weights.items()
    else:
        weights_iter = enumerate(hop_weights, start=1)
    weights: dict[int, float] = {}
    for hop, weight in weights_iter:
        hop_index = int(hop)
        if hop_index < 1:
            raise ValueError("hop indices must be positive integers")
        weight_value = float(weight)
        if weight_value != 0.0:
            weights[hop_index] = weight_value
    if not weights:
        raise ValueError("hop_weights must contain at least one non-zero weight")
    combined = np.zeros_like(base)
    max_hop = max(weights)
    current_power = base.copy()
    for hop in range(1, max_hop + 1):
        if hop > 1:
            current_power = current_power @ base
        if hop in weights:
            combined += weights[hop] * current_power
    if zero_diagonal:
        np.fill_diagonal(combined, 0.0)
    return combined


def population_mean_como(
    runner: "ABDRunner",
    previous_smu: np.ndarray,
    agent_index: int,
    time_index: int,
    shared_context: Mapping[str, Any],
) -> float:
    """Compare previous SMU for agent to population mean SMU."""

    mean_smu = shared_context["population_mean_smu"]
    return float(mean_smu - previous_smu[agent_index])


def population_mean_positive_como(
    runner: "ABDRunner",
    previous_smu: np.ndarray,
    agent_index: int,
    time_index: int,
    shared_context: Mapping[str, Any],
) -> float:
    """population_mean_como, but only positive values. No advantage in using SMU more than average."""

    mean_smu = shared_context["population_mean_smu"]
    diff = mean_smu - previous_smu[agent_index]
    return float(max(0.0, diff))


def population_mean_softplus_como(
    runner: "ABDRunner",
    previous_smu: np.ndarray,
    agent_index: int,
    time_index: int,
    shared_context: Mapping[str, Any],
    *,
    softness: float = 0.25,
) -> float:
    """Smoothly bounded variant of :func:`population_mean_positive_como`.

    The hard rectifier in :func:`population_mean_positive_como` introduces a
    kink that complicates analytical work.  Replacing it with a softplus
    produces a differentiable approximation to ``max(0, diff)`` while
    retaining the intuitive behaviour: COMO responds only when the population
    mean exceeds the agent's previous SMU.

    Parameters
    ----------
    softness:
        Controls how sharply the softplus transitions around zero.  Smaller
        values make the curve closely track ``max(0, diff)``; larger values
        spread the transition over a wider interval.
    """

    mean_smu = shared_context["population_mean_smu"]
    diff = mean_smu - previous_smu[agent_index]
    return float(_softplus(diff, softness))


def adjacency_weighted_como(
    runner: "ABDRunner",
    previous_smu: np.ndarray,
    agent_index: int,
    time_index: int,
    shared_context: Mapping[str, Any],
) -> float:
    """COMO based on adjacency-weighted neighbour averages.

    Requires an adjacency matrix to be supplied to :class:`ABDRunner`.
    """

    neighbour_means = shared_context["neighbour_mean_smu"]
    return float(neighbour_means[agent_index] - previous_smu[agent_index])


def adjacency_weighted_positive_como(
    runner: "ABDRunner",
    previous_smu: np.ndarray,
    agent_index: int,
    time_index: int,
    shared_context: Mapping[str, Any],
) -> float:
    """Adjacency COMO constrained to positive neighbour differences only.

    Requires an adjacency matrix to be supplied to :class:`ABDRunner`.
    """

    neighbour_means = shared_context["neighbour_mean_smu"]
    diff = neighbour_means[agent_index] - previous_smu[agent_index]
    return float(max(0.0, diff))


def adjacency_weighted_softplus_como(
    runner: "ABDRunner",
    previous_smu: np.ndarray,
    agent_index: int,
    time_index: int,
    shared_context: Mapping[str, Any],
    *,
    softness: float = 0.25,
) -> float:
    """Smooth analogue of :func:`adjacency_weighted_positive_como`.

    Applies the same softplus rectifier used by
    :func:`population_mean_softplus_como` to the adjacency-weighted neighbour
    deviations.  The ``softness`` parameter mirrors the population variant and
    lets users tune how sharply COMO reacts when neighbours exceed the
    agent's SMU.
    """

    neighbour_means = shared_context["neighbour_mean_smu"]
    diff = neighbour_means[agent_index] - previous_smu[agent_index]
    return float(_softplus(diff, softness))


def goldilocks_quadratic_como(
    runner: "ABDRunner",
    previous_smu: np.ndarray,
    agent_index: int,
    time_index: int,
    shared_context: Mapping[str, Any],
    *,
    width: float = 1.0,
) -> float:
    """Goldilocks COMO that penalises deviation from the population mean."""

    mean_smu = shared_context["population_mean_smu"]
    deviation = previous_smu[agent_index] - mean_smu
    return float((deviation / width) ** 2)


def adjacency_quadratic_como(
    runner: "ABDRunner",
    previous_smu: np.ndarray,
    agent_index: int,
    time_index: int,
    shared_context: Mapping[str, Any],
    *,
    width: float = 1.0,
) -> float:
    """Quadratic COMO based on adjacency-weighted neighbour averages."""

    neighbour_means = shared_context["neighbour_mean_smu"]
    deviation = previous_smu[agent_index] - neighbour_means[agent_index]
    return float((deviation / width) ** 2)


__all__ = [
    "COMOFunction",
    "build_grouped_adjacency",
    "build_ring_adjacency",
    "build_simple_adjacency",
    "population_mean_como",
    "population_mean_positive_como",
    "adjacency_weighted_como",
    "adjacency_weighted_positive_como",
    "population_mean_softplus_como",
    "adjacency_weighted_softplus_como",
    "goldilocks_quadratic_como",
    "adjacency_quadratic_como",
]
