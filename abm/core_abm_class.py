"""Core components for agent-based dynamics (ABD/ABM) simulations.

This module defines Python classes for describing agents, their decision rules, learning behaviour and
interaction structure (COMO and adjacency).  Adjacency utilities are defined in ``como_functions`` and
re-exported here for convenience so they can be plugged directly into :class:`ABDRunner`.

Example
-------
>>> from core_abm_class import (
...     AgentConfig,
...     BaseAgent,
...     adjacency_weighted_como,
...     ABDRunner,
...     build_ring_adjacency,
... )
>>> from decision_rules import AR1_decision
>>> from learning_rules import peer_learning
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> agents = [
...     BaseAgent(
...         config=AgentConfig(initial_wellbeing=8.0, initial_smu=3.0,
...                             parameters={"wb_baseline": 8.0, "smu_baseline": 3.0}),
...         decision_rule=AR1_decision(
...             smu_wb_coeff=-0.4, como_wb_coeff=-0.2,
...             smu_decay=0.7, smu_mean_reversion=0.3,
...             wellbeing_noise=0.4, smu_noise=0.6,
...         ),
...         learning_rule=peer_learning(
...             learning_rate=0.05, target="adjacency", direction="towards",
...         ),
...     )
...     for _ in range(5)
... ]
>>> adjacency = build_ring_adjacency(n_agents=5, neighbours=1)
>>> runner = ABDRunner(
...     agents=agents,
...     T=50,
...     como_function=adjacency_weighted_como,
...     adjacency_matrix=adjacency,
...     seed=0,
... )
>>> log = runner.run()
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from como_functions import (
    COMOFunction,
    build_grouped_adjacency,
    build_ring_adjacency,
    build_simple_adjacency,
    expand_adjacency_with_hops,
    adjacency_quadratic_como,
    adjacency_weighted_como,
    adjacency_weighted_positive_como,
    goldilocks_quadratic_como,
    population_mean_como,
    population_mean_positive_como,
)

# Dataclasses describing the moving parts


@dataclass
class AgentConfig:
    """Container for initial agent state and parameters.

    Parameters
    ----------
    initial_wellbeing:
        Wellbeing level used to seed the simulation at time ``t = 0``.
    initial_smu:
        Social media use value at ``t = 0``.
    parameters:
        Mutable parameters that the learning rules can update.  We keep them in
        a mapping so that notebook experiments can store arbitrary metadata
        (e.g. learning rates, aspiration levels, intrinsic wellbeing).
    label:
        Optional human readable identifier used in logging/plots.
    """

    initial_wellbeing: float
    initial_smu: float
    parameters: MutableMapping[str, float] = field(default_factory=dict)
    label: str | None = None
    group: str | None = None


@dataclass
class AgentState:
    """State snapshot for an agent at a single point in time."""

    wellbeing: float
    smu: float
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class AgentStepContext:
    """Information exposed to decision and learning rules each step."""

    time_index: int
    como: float
    population_mean_smu: float
    population_mean_wellbeing: float
    rng: np.random.Generator
    shared: Mapping[str, Any]
    neighbour_mean_smu: float | None = None
    group_mean_smu: float | None = None
    group_mean_wellbeing: float | None = None


@dataclass
class AgentStepResult:
    """Return value for a single agent update."""

    wellbeing: float
    smu: float
    logs: MutableMapping[str, float] = field(default_factory=dict)


@dataclass
class SimulationLog:
    """Structured output produced by :class:`ABDRunner`."""

    wellbeing: np.ndarray
    smu: np.ndarray
    como: np.ndarray
    parameter_history: dict[str, np.ndarray]
    extras: dict[str, np.ndarray]
    metadata: dict[str, Any]


# Agent abstraction

DecisionRule = Callable[["BaseAgent", AgentStepContext], AgentStepResult]
LearningRule = Callable[["BaseAgent", AgentStepContext, AgentStepResult], None]


class BaseAgent:
    """A flexible agent class.

    Each agent contains:
    - a current :class:`AgentState` with wellbeing and SMU values,
    - a mutable dictionary of parameters that learning rules can update,
    - a decision rule that converts the context into the next state,
    - an optional learning rule that adjusts parameters after each step.
    """

    def __init__(
        self,
        config: AgentConfig,
        decision_rule: DecisionRule,
        learning_rule: LearningRule | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.decision_rule = decision_rule
        self.learning_rule = learning_rule
        self.rng = np.random.default_rng() if rng is None else rng

        self.state = AgentState(
            wellbeing=config.initial_wellbeing,
            smu=config.initial_smu,
            extra={},
        )
        self.parameters: MutableMapping[str, float] = config.parameters
        self.label = config.label
        self.group = config.group

    # Hooks used by the simulation driver

    def step(self, context: AgentStepContext) -> AgentStepResult:
        """Advance the agent by one time step.

        The decision rule produces the next state; the learning rule can then
        mutate ``self.parameters`` with access to both the context and the
        freshly computed state.
        """

        result = self.decision_rule(self, context)

        # Ensure logs are mutable so learning rules can record diagnostics.
        result.logs = dict(result.logs)

        self.state = AgentState(
            wellbeing=result.wellbeing,
            smu=result.smu,
            extra=dict(result.logs),
        )

        if self.learning_rule is not None:
            self.learning_rule(self, context, result)
            self.state.extra = dict(result.logs)

        return result

    def snapshot_parameters(self) -> Mapping[str, float]:
        """Return parameters that should be logged.

        Subclasses can override to expose derived statistics; the base
        implementation simply returns ``self.parameters``.
        """

        return dict(self.parameters)


# Population helpers


def _compute_regression_diagnostics(
    wellbeing: np.ndarray, smu: np.ndarray, windows: Sequence[int | None]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling SMU->wellbeing regressions for diagnostics."""

    T_plus_one, n_agents = smu.shape
    slopes = np.full((T_plus_one, n_agents), np.nan)
    intercepts = np.full((T_plus_one, n_agents), np.nan)

    # Pre-compute cumulative sums so each window can be retrieved in O(1).
    prefix_smu = np.concatenate(
        (np.zeros((1, n_agents)), np.cumsum(smu, axis=0)), axis=0
    )
    prefix_wb = np.concatenate(
        (np.zeros((1, n_agents)), np.cumsum(wellbeing, axis=0)), axis=0
    )
    prefix_smu_wb = np.concatenate(
        (np.zeros((1, n_agents)), np.cumsum(smu * wellbeing, axis=0)), axis=0
    )
    prefix_smu_sq = np.concatenate(
        (np.zeros((1, n_agents)), np.cumsum(smu * smu, axis=0)), axis=0
    )

    for agent_idx, maybe_window in enumerate(windows):
        if maybe_window is None:
            continue

        window = max(int(maybe_window), 0)
        if window <= 1 or window > T_plus_one - 1:
            # Require at least two observations; fall back to NaNs otherwise.
            continue

        # Time indices for which the rolling window is valid.
        t_indices = np.arange(window, T_plus_one)
        if t_indices.size == 0:
            continue

        end = t_indices + 1
        start = end - window

        sum_x = prefix_smu[end, agent_idx] - prefix_smu[start, agent_idx]
        sum_y = prefix_wb[end, agent_idx] - prefix_wb[start, agent_idx]
        sum_xy = prefix_smu_wb[end, agent_idx] - prefix_smu_wb[start, agent_idx]
        sum_x2 = prefix_smu_sq[end, agent_idx] - prefix_smu_sq[start, agent_idx]

        window_f = float(window)
        denom = window_f * sum_x2 - sum_x * sum_x
        constant_mask = np.isclose(
            window_f * sum_x2,
            sum_x * sum_x,
            rtol=1e-05,
            atol=1e-08,
        )

        # Handle constant SMU windows: zero slope and mean wellbeing intercept.
        if np.any(constant_mask):
            mean_wb = sum_y[constant_mask] / window_f
            slopes[t_indices[constant_mask], agent_idx] = 0.0
            intercepts[t_indices[constant_mask], agent_idx] = mean_wb

        varying_mask = ~constant_mask
        if np.any(varying_mask):
            slope = (
                window_f * sum_xy[varying_mask]
                - sum_x[varying_mask] * sum_y[varying_mask]
            ) / denom[varying_mask]
            slope = np.clip(slope, -5.0, 5.0)
            intercept = (sum_y[varying_mask] - slope * sum_x[varying_mask]) / window_f

            slopes[t_indices[varying_mask], agent_idx] = slope
            intercepts[t_indices[varying_mask], agent_idx] = intercept

    return slopes, intercepts


def assign_learning_rules_by_fraction(
    agents: Sequence[BaseAgent],
    specifications: Sequence[tuple[str, Callable[[], LearningRule | None], float]],
    *,
    rng: np.random.Generator | None = None,
) -> Mapping[str, np.ndarray]:
    """Assign learning rules to agents according to fractional weights.

    Parameters
    ----------
    agents:
        Population of agents that will receive learning rules.  Their existing
        learning rules are overwritten.
    specifications:
        Sequence of ``(label, factory, fraction)`` tuples.  ``factory`` is a
        callable returning a :class:`LearningRule` (or ``None`` to disable
        learning) for each agent in the labelled cohort.  Fractions are
        normalised to sum to one and allocated using a largest-remainder method
        so the requested proportions are respected exactly.
    rng:
        Optional random generator controlling the cohort assignment.

    Returns
    -------
    Mapping[str, np.ndarray]
        Dictionary mapping each label to the array of agent indices assigned to
        the corresponding learning rule.  This is useful for post-processing and
        sanity checks in notebooks.
    """

    if not specifications:
        raise ValueError("At least one learning rule specification is required")

    n_agents = len(agents)
    if n_agents == 0:
        return {}

    labels, factories, fractions = zip(*specifications)

    fractions = np.array(fractions, dtype=float)
    if np.any(fractions < 0.0):
        raise ValueError("Learning rule fractions must be non-negative")
    total = fractions.sum()
    if total <= 0.0:
        raise ValueError("Sum of fractions must be positive")
    fractions = fractions / total

    expected_counts = fractions * n_agents
    counts = np.floor(expected_counts).astype(int)
    remainder = n_agents - counts.sum()
    if remainder > 0:
        fractional_parts = expected_counts - counts
        ordering = np.argsort(fractional_parts)[::-1]
        if ordering.size == 0:
            raise RuntimeError("No cohorts available for learning rule assignment")
        order_idx = 0
        while remainder > 0:
            counts[ordering[order_idx % ordering.size]] += 1
            remainder -= 1
            order_idx += 1

    rng = np.random.default_rng() if rng is None else rng
    assignment_indices = np.repeat(np.arange(len(specifications)), counts)
    if assignment_indices.size != n_agents:
        raise RuntimeError("Learning rule assignment did not allocate every agent")
    rng.shuffle(assignment_indices)

    label_to_indices: dict[str, list[int]] = {label: [] for label in labels}

    for idx, choice in enumerate(assignment_indices):
        label = labels[choice]
        rule_factory = factories[choice]
        learning_rule = None if rule_factory is None else rule_factory()
        agents[idx].learning_rule = learning_rule
        agents[idx].group = label
        agents[idx].config.group = label
        label_to_indices[label].append(idx)

    return {
        label: np.array(indices, dtype=int)
        for label, indices in label_to_indices.items()
    }


# Simulation driver


class ABDRunner:
    """Simulation harness coordinating a population of agents.

    Parameters
    ----------
    agents:
        Sequence of :class:`BaseAgent` instances.  They can be heterogeneous --
        different decision/learning rules work seamlessly.
    T:
        Number of time steps to simulate.
    como_function:
        Function converting past SMU values into COMO at each step.
    adjacency_matrix:
        Optional (n_agents, n_agents) array describing adjacency.
        Rows are row-normalised internally so callers can pass raw counts or
        binary matrices.
    shared_state_updater:
        Optional callable executed once per step to augment the shared context
        dictionary before agents are updated.  Use this hook to track custom
        statistics without subclassing the runner.
    seed:
        Random seed for reproducibility.  Each agent receives a fresh generator
        derived from this seed to keep notebooks deterministic.
    """

    def __init__(
        self,
        *,
        agents: Sequence[BaseAgent],
        T: int,
        como_function: COMOFunction = population_mean_como,
        adjacency_matrix: np.ndarray | None = None,
        shared_state_updater: (
            Callable[["ABDRunner", int, np.ndarray, np.ndarray, dict[str, Any]], None]
            | None
        ) = None,
        seed: int | None = None,
    ) -> None:
        if T <= 0:
            raise ValueError("T must be positive")

        self.agents = list(agents)
        self.T = int(T)
        self.como_function = como_function
        self.adjacency_matrix = (
            None
            if adjacency_matrix is None
            else np.array(adjacency_matrix, dtype=float)
        )
        self.shared_state_updater = shared_state_updater
        self.seed = seed

        if not self.agents:
            raise ValueError("At least one agent is required")

        self._rng = np.random.default_rng(seed)
        for idx, agent in enumerate(self.agents):
            agent.rng = np.random.default_rng(
                None if seed is None else self._rng.integers(0, 2**32 - 1)
            )

        if self.adjacency_matrix is not None:
            if self.adjacency_matrix.shape != (len(self.agents), len(self.agents)):
                raise ValueError("adjacency_matrix must match number of agents")
            self._normalised_adjacency = self._normalise_adjacency(
                self.adjacency_matrix
            )
        else:
            self._normalised_adjacency = None

        self.group_labels: list[str | None] = [agent.group for agent in self.agents]
        group_indices: dict[str, list[int]] = defaultdict(list)
        for idx, label in enumerate(self.group_labels):
            if label is not None:
                group_indices[label].append(idx)
        self._group_indices: dict[str, np.ndarray] = {
            label: np.array(indices, dtype=int)
            for label, indices in group_indices.items()
            if indices
        }

    # Utility helpers

    @staticmethod
    def _normalise_adjacency(adjacency: np.ndarray) -> np.ndarray:
        row_sums = adjacency.sum(axis=1, keepdims=True)
        normalised = np.where(row_sums > 0, adjacency / row_sums, 0.0)
        return normalised

    # Main execution entry point

    def run(self) -> SimulationLog:
        n_agents = len(self.agents)
        T = self.T

        wellbeing = np.zeros((T + 1, n_agents), dtype=float)
        smu = np.zeros((T + 1, n_agents), dtype=float)
        como = np.zeros((T + 1, n_agents), dtype=float)

        # Initialise state arrays with t = 0 values.
        for idx, agent in enumerate(self.agents):
            wellbeing[0, idx] = agent.state.wellbeing
            smu[0, idx] = agent.state.smu

        parameter_names = sorted(
            {key for agent in self.agents for key in agent.snapshot_parameters().keys()}
        )
        parameter_history: dict[str, np.ndarray] = {
            name: np.full((T + 1, n_agents), np.nan) for name in parameter_names
        }
        for idx, agent in enumerate(self.agents):
            params = agent.snapshot_parameters()
            for name in parameter_names:
                if name in params:
                    parameter_history[name][0, idx] = params[name]

        extras: dict[str, np.ndarray] = {}

        shared_context: dict[str, Any] = {}

        for t in range(1, T + 1):
            previous_wellbeing = wellbeing[t - 1]
            previous_smu = smu[t - 1]

            population_mean_wb = float(previous_wellbeing.mean())
            population_mean_smu = float(previous_smu.mean())

            neighbour_means = None
            if self._normalised_adjacency is not None:
                neighbour_means = self._normalised_adjacency @ previous_smu

            group_mean_wb: dict[str, float] | None = None
            group_mean_smu: dict[str, float] | None = None
            if self._group_indices:
                group_mean_wb = {}
                group_mean_smu = {}
                for label, indices in self._group_indices.items():
                    group_mean_wb[label] = float(previous_wellbeing[indices].mean())
                    group_mean_smu[label] = float(previous_smu[indices].mean())

            shared_context.update(
                {
                    "population_mean_wellbeing": population_mean_wb,
                    "population_mean_smu": population_mean_smu,
                    "neighbour_mean_smu": neighbour_means,
                    "group_mean_wellbeing": group_mean_wb,
                    "group_mean_smu": group_mean_smu,
                    "time_index": t,
                }
            )

            if self.shared_state_updater is not None:
                self.shared_state_updater(
                    self, t, previous_wellbeing, previous_smu, shared_context
                )

            for idx, agent in enumerate(self.agents):
                group_label = self.group_labels[idx]
                agent_group_mean_wb = (
                    None
                    if group_label is None or group_mean_wb is None
                    else group_mean_wb[group_label]
                )
                agent_group_mean_smu = (
                    None
                    if group_label is None or group_mean_smu is None
                    else group_mean_smu[group_label]
                )

                agent_context = AgentStepContext(
                    time_index=t,
                    como=self.como_function(self, previous_smu, idx, t, shared_context),
                    population_mean_smu=population_mean_smu,
                    population_mean_wellbeing=population_mean_wb,
                    neighbour_mean_smu=(
                        None if neighbour_means is None else float(neighbour_means[idx])
                    ),
                    group_mean_smu=agent_group_mean_smu,
                    group_mean_wellbeing=agent_group_mean_wb,
                    rng=agent.rng,
                    shared=shared_context,
                )

                result = agent.step(agent_context)

                wellbeing[t, idx] = result.wellbeing
                smu[t, idx] = result.smu
                como[t, idx] = agent_context.como

                for key, value in result.logs.items():
                    if key not in extras:
                        extras[key] = np.full((T + 1, n_agents), np.nan)
                    extras[key][t, idx] = value

                params = agent.snapshot_parameters()
                for name in parameter_names:
                    if name in params:
                        parameter_history[name][t, idx] = params[name]

        metadata = {
            "time_index": np.arange(T + 1),
            "agent_labels": [agent.label for agent in self.agents],
            "group_labels": self.group_labels,
            "T": T,
        }

        needs_regression_logs = [
            getattr(agent.learning_rule, "_records_regression_diagnostics", False)
            for agent in self.agents
        ]
        if any(needs_regression_logs) and "regression_slope" not in extras:
            windows = [
                (
                    int(getattr(agent.learning_rule, "_regression_window", 10))
                    if flag and agent.learning_rule is not None
                    else None
                )
                for agent, flag in zip(self.agents, needs_regression_logs)
            ]

            slopes_subset, intercepts_subset = _compute_regression_diagnostics(
                wellbeing, smu, windows
            )

            extras["regression_slope"] = slopes_subset
            if "regression_intercept" not in extras:
                extras["regression_intercept"] = intercepts_subset
            if "regression_window" not in extras:
                window_array = np.full((T + 1, n_agents), np.nan)
                for agent_idx, maybe_window in enumerate(windows):
                    if maybe_window is not None:
                        window_array[:, agent_idx] = float(maybe_window)
                extras["regression_window"] = window_array

        return SimulationLog(
            wellbeing=wellbeing,
            smu=smu,
            como=como,
            parameter_history=parameter_history,
            extras=extras,
            metadata=metadata,
        )


__all__ = [
    "ABDRunner",
    "AgentConfig",
    "AgentState",
    "AgentStepContext",
    "AgentStepResult",
    "BaseAgent",
    "COMOFunction",
    "SimulationLog",
    "adjacency_quadratic_como",
    "adjacency_weighted_como",
    "adjacency_weighted_positive_como",
    "assign_learning_rules_by_fraction",
    "build_grouped_adjacency",
    "build_ring_adjacency",
    "build_simple_adjacency",
    "expand_adjacency_with_hops",
    "goldilocks_quadratic_como",
    "population_mean_como",
    "population_mean_positive_como",
]
