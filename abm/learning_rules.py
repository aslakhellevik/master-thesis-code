"""Learning rules.

This file contains the learning rules for agents that can be used in the simulations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core_abm_class import AgentStepContext, AgentStepResult, BaseAgent

LearningRule = Callable[[BaseAgent, AgentStepContext, AgentStepResult], None]

__all__ = [
    "LearningRule",
    "regression_based_learning",
    "gradient_ascent_learning",
    "peer_learning",
]


def regression_based_learning(*, learning_rate: float) -> LearningRule:
    """Learning rule based on slope of regressing WB on 10 last SMU values.

    Parameters
    ----------
    learning_rate:
        Step size for SMU target adjustments.  Clamped to ``[0, 0.2]`` to
        prevent overshooting.
    """

    base_learning_rate = float(min(0.2, max(0.0, learning_rate)))

    def _history_list(agent: BaseAgent, attr: str) -> list[float]:
        history = getattr(agent, attr, None)
        if history is None:
            history = []
            setattr(agent, attr, history)
        return history

    window = 10

    def _learn(
        agent: BaseAgent, context: AgentStepContext, result: AgentStepResult
    ) -> None:
        history_wb = _history_list(agent, "history_wb")
        history_smu = _history_list(agent, "history_smu")
        history_wb.append(float(result.wellbeing))
        history_smu.append(float(result.smu))

        if len(history_smu) < window:
            result.logs.setdefault("regression_slope", float("nan"))
            result.logs.setdefault("regression_intercept", float("nan"))
            result.logs.setdefault("regression_window", float(window))
            return

        X_window = np.array(history_smu[-window:], dtype=float)
        y_window = np.array(history_wb[-window:], dtype=float)
        if np.allclose(X_window, X_window[0]):
            slope = 0.0
            intercept = float(y_window.mean())
        else:
            design = np.column_stack([X_window, np.ones(window)])
            coef, *_ = np.linalg.lstsq(design, y_window, rcond=None)
            slope = float(coef[0])
            intercept = float(coef[1])
        slope = float(np.clip(slope, -5.0, 5.0))
        intercept = float(intercept)

        lr = float(agent.parameters.get("learning_rate", base_learning_rate))
        lr = float(min(0.2, max(0.0, lr)))
        delta = float(np.clip(lr * slope, -1.0, 1.0))

        smu_mu = float(agent.parameters.get("smu_mu", result.smu))
        smu_mu = float(np.clip(smu_mu + delta, 1.0, 15.0))

        agent.parameters["smu_mu"] = smu_mu

        result.logs["regression_slope"] = slope
        result.logs["regression_intercept"] = intercept
        result.logs["regression_window"] = float(window)

    # Mark the learning rule so the simulator can recover diagnostics when
    # running outside the notebooks (e.g. in automated scripts).
    _learn._records_regression_diagnostics = True  # type: ignore[attr-defined]
    _learn._regression_window = window  # type: ignore[attr-defined]

    return _learn


def gradient_ascent_learning(
    *,
    learning_rate: float,
    parameter: str = "smu_baseline",
    max_step: float = 1.0,
) -> LearningRule:
    """Learning rule based on approximate wellbeing gradient."""

    learning_rate = float(max(0.0, learning_rate))
    max_step = float(max(0.0, max_step))

    def _learn(
        agent: BaseAgent, context: AgentStepContext, result: AgentStepResult
    ) -> None:
        previous_wb = agent.parameters.get("_grad_prev_wb")
        previous_smu = agent.parameters.get("_grad_prev_smu")

        current_param = float(agent.parameters.get(parameter, result.smu))

        if previous_wb is not None and previous_smu is not None:
            delta_wb = float(result.wellbeing) - float(previous_wb)
            delta_smu = float(result.smu) - float(previous_smu)
            if abs(delta_smu) > 1e-6 and learning_rate > 0.0:
                gradient = delta_wb / delta_smu
                step = np.clip(learning_rate * gradient, -max_step, max_step)
                agent.parameters[parameter] = current_param + float(step)

        agent.parameters["_grad_prev_wb"] = float(result.wellbeing)
        agent.parameters["_grad_prev_smu"] = float(result.smu)

    return _learn


def peer_learning(
    *,
    learning_rate: float,
    parameter: str = "smu_baseline",
    target: str = "population",
    direction: str = "towards",
) -> LearningRule:
    """Move an agent towards or away from their peers' SMU target."""

    learning_rate = float(max(0.0, learning_rate))

    if target not in {"adjacency", "population"}:
        raise ValueError("target must be 'adjacency' or 'population'")
    if direction not in {"towards", "away"}:
        raise ValueError("direction must be 'towards' or 'away'")

    def _learn(
        agent: BaseAgent, context: AgentStepContext, result: AgentStepResult
    ) -> None:
        if learning_rate == 0.0:
            return

        if target == "adjacency":
            if context.neighbour_mean_smu is None:
                return
            peer_value = float(context.neighbour_mean_smu)
        else:
            peer_value = float(context.population_mean_smu)

        current_param = float(agent.parameters.get(parameter, result.smu))

        if direction == "towards":
            updated = current_param + learning_rate * (peer_value - current_param)
        else:
            updated = current_param + learning_rate * (current_param - peer_value)

        agent.parameters[parameter] = float(updated)

    return _learn
