"""Decision rules.

This file contains the decision rules for agents that can be used in the simulations.
"""

from __future__ import annotations

from collections.abc import Callable, MutableSequence

import numpy as np

from core_abm_class import AgentStepContext, AgentStepResult, BaseAgent

DecisionRule = Callable[[BaseAgent, AgentStepContext], AgentStepResult]


__all__ = [
    "DecisionRule",
    "AR1_decision",
    "default_decision",
    "smooth_smu_with_spillover_decision",
]

# Helper functions


def _ensure_delay_queue(agent: BaseAgent) -> MutableSequence[float]:
    """Return the delayed decline queue attached to the agent."""

    queue = getattr(agent, "delayed_effects", None)
    if queue is None:
        queue = []
        setattr(agent, "delayed_effects", queue)
    return queue


def _clip(value: float, bounds: tuple[float, float]) -> float:
    """Smooth clipping into bounds using a tanh squash."""

    lower, upper = (float(b) for b in bounds)
    value = float(value)

    span = upper - lower
    mid = lower + 0.5 * span

    # Choose the "temperature" so that the derivative at the midpoint is around 1 for near-linearity
    temperature = max(span / 4.0, np.finfo(float).tiny)
    scaled = (value - mid) / temperature
    squashed = 0.5 * (1.0 + np.tanh(0.5 * scaled))

    return float(lower + span * squashed)


def _ensure_named_queue(agent: BaseAgent, attribute: str) -> MutableSequence[float]:
    """Return a mutable queue stored on ``agent`` under ``attribute``."""

    queue = getattr(agent, attribute, None)
    if queue is None:
        queue = []
        setattr(agent, attribute, queue)
    return queue


def _difference_of_exponentials(
    *,
    horizon: int,
    positive_scale: float,
    negative_scale: float,
    fast_timescale: float,
    slow_timescale: float,
    target_total_effect: float | None = None,
) -> np.ndarray:
    """Return a smooth "boost then decline" curve."""

    horizon = max(1, int(horizon))
    fast_timescale = max(1e-6, float(fast_timescale))
    slow_timescale = max(1e-6, float(slow_timescale))

    t = np.arange(horizon, dtype=float)
    positive = np.exp(-t / fast_timescale)
    negative = np.exp(-t / slow_timescale)
    curve = positive_scale * positive - negative_scale * negative

    if target_total_effect is not None:
        current = float(curve.sum())
        curve += (float(target_total_effect) - current) / horizon

    return curve


def _accumulate_queue(queue: MutableSequence[float], values: np.ndarray) -> None:
    """Add ``values`` into ``queue`` element-wise, extending as required."""

    required = len(values)
    while len(queue) < required:
        queue.append(0.0)
    for idx, value in enumerate(values):
        queue[idx] += float(value)


def AR1_decision(
    *,
    smu_wb_coeff: float,
    como_wb_coeff: float,
    smu_decay: float,
    smu_mean_reversion: float,
    wellbeing_noise: float,
    smu_noise: float,
    wellbeing_bounds: tuple[float, float] = (float("-inf"), float("inf")),
    smu_bounds: tuple[float, float] = (float("-inf"), float("inf")),
) -> DecisionRule:
    """Implements an AR(1) decision rule.

    Wellbeing is updated as a linear function of the agent's own SMU and COMO.
    SMU follows an AR(1) process with optional mean reversion towards the population average.
    """

    smu_bounds = tuple(float(b) for b in smu_bounds)
    wellbeing_bounds = tuple(float(b) for b in wellbeing_bounds)

    def _rule(agent: BaseAgent, context: AgentStepContext) -> AgentStepResult:
        rng = context.rng

        wb_baseline = agent.parameters.get("wb_baseline", agent.state.wellbeing)
        smu_baseline = agent.parameters.get("smu_baseline", agent.state.smu)

        smu = (
            smu_baseline * (1.0 - smu_mean_reversion)
            + smu_mean_reversion * context.population_mean_smu
            + smu_decay * agent.state.smu
            + rng.normal(0.0, smu_noise)
        )
        smu = _clip(smu, smu_bounds)

        wellbeing = (
            wb_baseline
            + smu_wb_coeff * agent.state.smu
            + como_wb_coeff * context.como
            + rng.normal(0.0, wellbeing_noise)
        )
        wellbeing = _clip(wellbeing, wellbeing_bounds)

        logs = {
            "wb_baseline": wb_baseline,
            "smu_baseline": smu_baseline,
        }
        return AgentStepResult(wellbeing=wellbeing, smu=smu, logs=logs)

    return _rule


def default_decision(
    *,
    rho: float,
    smu_wb_coeff: float,
    como_wb_coeff: float,
    delayed_decline: str = "False",
    boost_coeff: float = 0.0,
    decline_coeff: float = 0.0,
    decline_lag: int = 1,
    reward_scale: float = 0.0,
    penalty_scale: float = 0.0,
    fast_timescale: float = 1.0,
    slow_timescale: float = 1.0,
    effect_horizon: int = 1,
    target_total_effect: float | None = None,
    como_delayed_decline: str = "False",
    como_boost_coeff: float = 0.0,
    como_decline_coeff: float = 0.0,
    como_decline_lag: int = 1,
    como_reward_scale: float = 0.0,
    como_penalty_scale: float = 0.0,
    como_fast_timescale: float = 1.0,
    como_slow_timescale: float = 1.0,
    como_effect_horizon: int | None = None,
    como_target_total_effect: float | None = None,
    smu_bounds: tuple[float, float] = (0.0, 16.0),
    wellbeing_bounds: tuple[float, float] = (0.0, 10.0),
    use_hard_clip: bool = False,
) -> DecisionRule:
    """Decision rule allowing for multiple configurations.

    Draws SMU/WB from normal distributions, incorporates COMO, and optionally a boost and
    delayed decline effect.  Two separate delayed-effect pipelines can be configured:

    ``delayed_decline`` controls the SMU-driven queue, while ``como_delayed_decline``
    applies the same mechanics to COMO.  Each parameter accepts the same modes:

    "False": No delayed decline effect.
    "simple": Apply a boost-plus-linear-decline queue.
    "smooth": Apply a smooth boost/penalty curve queue.

    Parameters
    ----------
    use_hard_clip : bool
        If True, use hard clipping (np.clip) instead of smooth tanh-based clipping.
        Hard clipping preserves linearity for estimation. Default is False (smooth clipping).
    """

    rho = float(rho)
    decline_lag = int(decline_lag)
    smu_bounds = tuple(float(b) for b in smu_bounds)
    wellbeing_bounds = tuple(float(b) for b in wellbeing_bounds)

    decline_mode = str(delayed_decline).lower()

    use_simple_decline = decline_mode == "simple"

    if decline_mode == "smooth":
        boost_curve = _difference_of_exponentials(
            horizon=effect_horizon,
            positive_scale=float(reward_scale),
            negative_scale=float(penalty_scale),
            fast_timescale=float(fast_timescale),
            slow_timescale=float(slow_timescale),
            target_total_effect=target_total_effect,
        )

    como_decline_mode = str(como_delayed_decline).lower()
    como_use_simple_decline = como_decline_mode == "simple"

    if como_effect_horizon is None:
        como_effect_horizon = effect_horizon

    if como_decline_mode == "smooth":
        como_boost_curve = _difference_of_exponentials(
            horizon=como_effect_horizon,
            positive_scale=float(como_reward_scale),
            negative_scale=float(como_penalty_scale),
            fast_timescale=float(como_fast_timescale),
            slow_timescale=float(como_slow_timescale),
            target_total_effect=como_target_total_effect,
        )

    def _rule(agent: BaseAgent, context: AgentStepContext) -> AgentStepResult:
        rng = context.rng

        smu_mu = float(agent.parameters.get("smu_mu", agent.state.smu))
        smu_sigma = float(agent.parameters.get("smu_sigma", 1.0))
        smu_sigma = max(smu_sigma, 1e-6)

        last_smu = float(agent.state.smu)
        epsilon_smu = float(rng.normal(0, smu_sigma))
        smu = rho * last_smu + (1.0 - rho) * smu_mu + epsilon_smu
        if use_hard_clip:
            smu = float(np.clip(smu, smu_bounds[0], smu_bounds[1]))
        else:
            smu = _clip(smu, smu_bounds)

        wb_mu = float(agent.parameters.get("wb_mu", agent.state.wellbeing))
        wb_sigma = float(agent.parameters.get("wb_sigma", 1.0))
        wb_sigma = max(wb_sigma, 1e-6)
        base_wb = float(rng.normal(wb_mu, wb_sigma))

        wellbeing = base_wb + smu_wb_coeff * smu + como_wb_coeff * context.como

        if use_simple_decline:
            queue = _ensure_delay_queue(agent)
            wellbeing += boost_coeff * smu
            total_decline = decline_coeff * smu
            lag = max(1, decline_lag)
            daily_decline = total_decline / lag

            while len(queue) < lag:
                queue.append(0.0)
            for i in range(lag):
                queue[i] += daily_decline
            if queue:
                wellbeing -= queue.pop(0)
        elif decline_mode == "smooth":
            queue = _ensure_named_queue(agent, "smooth_effects_queue")
            instantaneous_effect = 0.0
            if queue:
                instantaneous_effect = float(queue.pop(0))
            wellbeing += instantaneous_effect

            _accumulate_queue(queue, boost_curve * smu)

        if como_use_simple_decline:
            como_queue = _ensure_named_queue(agent, "como_delayed_effects")
            wellbeing += como_boost_coeff * context.como
            total_como_decline = como_decline_coeff * context.como
            como_lag = max(1, int(como_decline_lag))
            daily_como_decline = total_como_decline / como_lag

            while len(como_queue) < como_lag:
                como_queue.append(0.0)
            for i in range(como_lag):
                como_queue[i] += daily_como_decline
            if como_queue:
                wellbeing -= como_queue.pop(0)
        elif como_decline_mode == "smooth":
            como_queue = _ensure_named_queue(agent, "como_smooth_effects_queue")
            como_instantaneous_effect = 0.0
            if como_queue:
                como_instantaneous_effect = float(como_queue.pop(0))
            wellbeing += como_instantaneous_effect

            _accumulate_queue(como_queue, como_boost_curve * context.como)

        # Store unclipped wellbeing for RCT adjustment (before clipping distorts the linear relationship)
        wellbeing_unclipped = wellbeing

        if use_hard_clip:
            wellbeing = float(
                np.clip(wellbeing, wellbeing_bounds[0], wellbeing_bounds[1])
            )
        else:
            wellbeing = _clip(wellbeing, wellbeing_bounds)

        logs = {
            "smu_target": smu_mu,
            "smu_sigma": smu_sigma,
            "wellbeing_unclipped": wellbeing_unclipped,
        }
        if decline_mode == "smooth":
            logs["smooth_effect"] = instantaneous_effect
        if como_decline_mode == "smooth":
            logs["como_smooth_effect"] = como_instantaneous_effect
        return AgentStepResult(wellbeing=wellbeing, smu=smu, logs=logs)

    return _rule


def smooth_smu_with_spillover_decision(
    *,
    rho: float,
    smu_wb_coeff: float,
    como_wb_coeff: float,
    reward_scale: float,
    penalty_scale: float,
    fast_timescale: float,
    slow_timescale: float,
    effect_horizon: int,
    spillover_scale: float,
    spillover_timescale: float,
    spillover_horizon: int | None = None,
    target_total_effect: float | None = None,
    spillover_target_total_effect: float | None = None,
    smu_bounds: tuple[float, float] = (0.0, 16.0),
    wellbeing_bounds: tuple[float, float] = (0.0, 10.0),
) -> DecisionRule:
    """Decision rule with smooth delayed decline and neighbour spillover.

    Extends the smooth-decline pipeline from :func:`default_decision` by
    distributing a fraction of each agent's delayed effect to adjacent
    agents, weighted by ``neighbour_weights``.  The spillover curve is a
    decaying exponential whose magnitude and timescale are controlled
    independently of the agent's own boost/penalty curve.
    """

    rho = float(rho)
    smu_bounds = tuple(float(b) for b in smu_bounds)
    wellbeing_bounds = tuple(float(b) for b in wellbeing_bounds)

    boost_curve = _difference_of_exponentials(
        horizon=effect_horizon,
        positive_scale=float(reward_scale),
        negative_scale=float(penalty_scale),
        fast_timescale=float(fast_timescale),
        slow_timescale=float(slow_timescale),
        target_total_effect=target_total_effect,
    )

    spillover_horizon = (
        effect_horizon if spillover_horizon is None else spillover_horizon
    )
    spillover_curve = -abs(spillover_scale) * np.exp(
        -np.arange(max(1, int(spillover_horizon)), dtype=float)
        / max(1e-6, float(spillover_timescale))
    )
    if spillover_target_total_effect is not None:
        current = float(spillover_curve.sum())
        if abs(current) > 1e-9:
            spillover_curve *= float(spillover_target_total_effect) / current

    def _rule(agent: BaseAgent, context: AgentStepContext) -> AgentStepResult:
        rng = context.rng

        smu_mu = float(agent.parameters.get("smu_mu", agent.state.smu))
        smu_sigma = float(agent.parameters.get("smu_sigma", 1.0))
        smu_sigma = max(smu_sigma, 1e-6)

        last_smu = float(agent.state.smu)
        epsilon_smu = float(rng.normal(0, smu_sigma))
        smu = rho * last_smu + (1.0 - rho) * smu_mu + epsilon_smu
        smu = _clip(smu, smu_bounds)

        wb_mu = float(agent.parameters.get("wb_mu", agent.state.wellbeing))
        wb_sigma = float(agent.parameters.get("wb_sigma", 1.0))
        wb_sigma = max(wb_sigma, 1e-6)
        base_wb = float(rng.normal(wb_mu, wb_sigma))

        wellbeing = base_wb + smu_wb_coeff * smu + como_wb_coeff * context.como

        queue = _ensure_named_queue(agent, "smooth_effects_queue")
        instantaneous_effect = 0.0
        if queue:
            instantaneous_effect += float(queue.pop(0))

        spillover_map = context.shared.setdefault("smooth_spillover_queues", {})
        agent_index = getattr(agent, "agent_index", None)
        if agent_index is not None:
            neighbour_queue = spillover_map.get(agent_index)
            if neighbour_queue:
                instantaneous_effect += float(neighbour_queue.pop(0))
                if not neighbour_queue:
                    spillover_map.pop(agent_index, None)

        wellbeing += instantaneous_effect

        _accumulate_queue(queue, boost_curve * smu)

        intensity = smu
        neighbours = getattr(agent, "neighbour_weights", None)
        if neighbours:
            for neighbour_idx, weight in neighbours:
                if weight <= 0:
                    continue
                neighbour_queue = spillover_map.setdefault(int(neighbour_idx), [])
                _accumulate_queue(
                    neighbour_queue,
                    spillover_curve * intensity * float(weight),
                )

        wellbeing = _clip(wellbeing, wellbeing_bounds)

        logs = {
            "smu_target": smu_mu,
            "smu_sigma": smu_sigma,
            "smooth_effect": instantaneous_effect,
        }
        return AgentStepResult(wellbeing=wellbeing, smu=smu, logs=logs)

    return _rule
